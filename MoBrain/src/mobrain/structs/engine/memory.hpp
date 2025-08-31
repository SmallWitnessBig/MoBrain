//
// Created by 31530 on 2025/8/13.
//

#ifndef MOBRAIN_MEMORY_HPP
#define MOBRAIN_MEMORY_HPP
#include "graphics.hpp"
#include "../scene/scene.hpp"
#include <vulkan/vulkan_raii.hpp>

#include <unordered_map>


namespace std {
    template<> struct hash<InstanceType> {
        size_t operator()(const InstanceType& type) const {
            return static_cast<size_t>(type);
        }
    };
}

class BufferManager {
public:
    // 缓冲区块大小（可配置）
    // 减小默认块大小以减少内存使用，避免OutOfDeviceMemory错误
    static constexpr vk::DeviceSize VERTEX_BLOCK_SIZE = 32 * 1024 * 1024;
    static constexpr vk::DeviceSize INDEX_BLOCK_SIZE = 32 * 1024 * 1024;
    static constexpr vk::DeviceSize INSTANCE_BLOCK_SIZE = 32 * 1024 * 1024;

    // 缓冲区块结构
    struct BufferBlock {
        vk::raii::Buffer buffer{ nullptr };
        vk::raii::DeviceMemory memory{ nullptr };
        vk::DeviceSize used = 0;
        vk::DeviceSize capacity = 0;
    };

    // 顶点和索引缓冲区块
    std::vector<BufferBlock> vertexBlocks;
    std::vector<BufferBlock> indexBlocks;
    std::vector<BufferBlock> instanceBlocks;

    std::vector<uint32_t>    freeVertexBlockIndices;
    std::vector<uint32_t>    freeIndexBlockIndices;
    std::vector<uint32_t>    freeInstanceBlockIndices;

    // 对象在缓冲区中的位置信息
    struct ObjectLocation {
        uint32_t vertexBlockIndex;  // 所在块索引
        uint32_t indexBlockIndex;
        uint32_t vertexOffset; // 在块内的偏移
        uint32_t indexOffset;
    };

    struct InstanceLocation {
        uint32_t instanceBlockIndex;
        uint32_t instanceOffset;
    };

    struct InstanceFirstLocation {
        uint32_t vertexBlockIndex;  // 所在块索引
        uint32_t indexBlockIndex;
        uint32_t vertexOffset; // 在块内的偏移
        uint32_t indexOffset;
    };

    std::unordered_map<const Object*, ObjectLocation> objectLocations;

    std::unordered_map<const Cube*, InstanceLocation> instanceLocations;

    std::unordered_map<InstanceType, InstanceFirstLocation> instanceFirstLocations;

    BufferManager& init();
    void initCubeInstance();

    // 添加对象到缓冲区
    void addObject(const Object* obj);
    void addCube(const Cube* cube);

    void removeCube(const Cube* cube);


    void updateObject(const Object* obj);


private:
    // 分配新的顶点块
    uint32_t allocateVertexBlock();

    // 分配新的索引块
    uint32_t allocateIndexBlock();

    uint32_t allocateInstanceBlock();

    // 上传数据到指定块的指定位置
    void uploadData(BufferBlock& block, const void* data, size_t size, uint32_t offset);

};
#endif //MOBRAIN_MEMORY_HPP