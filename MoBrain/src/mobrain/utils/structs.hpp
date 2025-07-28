#pragma once

#include <cstddef>
#include <array>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <queue>
#include <functional>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLFW_INCLUDE_NONE
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;


    static vk::VertexInputBindingDescription getBindingDescription();

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions();
};

struct UniformBufferObject{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

enum class ObjectType{
    CUBE
};

// Object 一个抽象类，定义一些基本的属性
class Object{
public:
    uint32_t index_count = 0;
    glm::vec3 pos;
    glm::vec3 color;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    Object(const Object& _o) =delete;
    Object& operator=(const Object& _o) =delete;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;
    Object(glm::vec3 _pos, glm::vec3 _color) :pos(_pos), color(_color) {};
    virtual ~Object() = default;
    virtual Object& draw(const vk::raii::CommandBuffer& _cmb) = 0;
    virtual ObjectType getType() = 0;
};
//Cube 
//1.创建cube
//2.存入缓存
//3.绘制
class Cube : public Object {
public:
    Cube(glm::vec3 _pos, glm::vec3 _color);
    Cube(const Cube&) = delete;
    Cube& operator=(const Cube&) = delete;
    Cube(Cube&&) = default;
    Cube& operator=(Cube&&) = default;
    ~Cube() = default;
    Cube& draw(const vk::raii::CommandBuffer& _cmb) override;
    ObjectType getType() override;
};
struct Vec3Hash {
    std::size_t operator()(const glm::vec3& v) const {
        std::size_t h1 = std::hash<float>{}(v.x);
        std::size_t h2 = std::hash<float>{}(v.y);
        std::size_t h3 = std::hash<float>{}(v.z);
        return ((h1 ^ (h2 << 1)) >> 1) ^ (h3 << 1);
    }
};
class Scene {
public:
    std::vector<std::shared_ptr<Object>> objects;
    std::vector<glm::vec3>               positions;
//    std::unordered_map<glm::vec3,std::vector<std::shared_ptr<Object>>,Vec3Hash> data;
//    std::vector<std::shared_ptr<Object>> getObjects(glm::vec3 _p);
    Scene& addObject(std::shared_ptr<Object> _o);
    Scene& removeObject(std::shared_ptr<Object> _o);
    Scene& Scene::removeObject(glm::vec3 _pos);
};
class Role {
public:
    std::vector<Cube> reach;
    Role& update();
};

class BufferManager {
public:
    // 缓冲区块大小（可配置）
    // 减小默认块大小以减少内存使用，避免OutOfDeviceMemory错误
    static constexpr vk::DeviceSize VERTEX_BLOCK_SIZE = 2 * 1024 * 1024; // 2MB (原为4MB)
    static constexpr vk::DeviceSize INDEX_BLOCK_SIZE = 1 * 1024 * 1024;  // 1MB (原为2MB)

    // 缓冲区块结构
    struct BufferBlock {
        vk::raii::Buffer buffer{ nullptr };
        vk::raii::DeviceMemory memory{ nullptr };
        vk::DeviceSize used = 0;
        vk::DeviceSize capacity;
    };

    // 顶点和索引缓冲区块
    std::vector<BufferBlock> vertexBlocks;
    std::vector<BufferBlock> indexBlocks;

    // 对象在缓冲区中的位置信息
    struct ObjectLocation {
        uint32_t vertexBlockIndex;  // 所在块索引
        uint32_t indexBlockIndex;
        uint32_t vertexOffset; // 在块内的偏移
        uint32_t indexOffset;
    };

    // 缓冲区块
    struct BlockSpace {
        uint32_t       blockIndex;
        vk::DeviceSize freeSpace;

        bool operator<(const BlockSpace& rhs) const {
            return freeSpace > rhs.freeSpace;
        }
    };

    std::priority_queue<BlockSpace> vertexSpaceIndex;
    std::priority_queue<BlockSpace> indexSpaceIndex;

    std::unordered_map<std::shared_ptr<Object>, ObjectLocation> objectLocations;

    // 初始化缓冲区管理器
    void initialize();

    // 添加对象到缓冲区
    void addObject(std::shared_ptr<Object> obj);

    // 更新对象数据
    void updateObject(std::shared_ptr<Object> obj);

private:
    // 分配新的顶点块
    uint32_t allocateVertexBlock();

    // 分配新的索引块
    uint32_t allocateIndexBlock();
    // 上传数据到指定块的指定位置
    void uploadData(BufferBlock& block, const void* data, size_t size, uint32_t offset);

};