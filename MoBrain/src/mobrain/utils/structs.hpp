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
#define MIN_OBJECT_VERTICES_COUNT  8
#define MIN_OBJECT_INDICES_COUNT  36
// 顶点数据
struct Vertex {
    glm::vec3 pos;
    static vk::VertexInputBindingDescription getBindingDescription() {
        return vk::VertexInputBindingDescription{
            0,
            sizeof(Vertex),
            vk::VertexInputRate::eVertex
        };
    };

    static vk::VertexInputAttributeDescription getAttributeDescriptions() {
        return vk::VertexInputAttributeDescription{
            0,
            0,
            vk::Format::eR32G32B32A32Sfloat,
            offsetof(Vertex,pos)
        };
    };
};

// 实例数据
class InstanceData {
public:
    glm::mat4 model;
    glm::vec4 color;

    static vk::VertexInputBindingDescription getBindingDescription() {
         return vk::VertexInputBindingDescription{
            1,
            sizeof(InstanceData),
            vk::VertexInputRate::eInstance
         };
    }

    static std::array<vk::VertexInputAttributeDescription, 5>  getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 5> attributeDescriptions;
        for (uint32_t i = 0; i < 5; ++i) {
            attributeDescriptions[i].binding = 1; // binding 1 for instance data
            attributeDescriptions[i].location = 1 + i; // location 1,2,3,4,5
            attributeDescriptions[i].format = vk::Format::eR32G32B32A32Sfloat;
            attributeDescriptions[i].offset = sizeof(glm::vec4) * i;
        }
        return attributeDescriptions;
    }

};


struct UniformBufferObject{
    glm::mat4 view;
    glm::mat4 proj;
};


// Object 一个抽象类，定义一些基本的属性
class Object{
public:
    uint32_t index_count = 0;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    Object(const Object& _o) = default;
    Object& operator=(const Object& _o) = default;
    Object(Object&&) = default;
    Object& operator=(Object&&) = default;
    virtual ~Object() = default;
};



//Cube 

class Cube : public InstanceData {
public:
    glm::vec3 pos;
    Cube(glm::vec3 _pos, glm::vec4 _color) : pos(_pos){
        color = _color;
        model = glm::translate(glm::mat4{ 1.0f }, _pos);
    };
    Cube(const Cube& c) {
        color = c.color;
        model = c.model;
        pos = c.pos;
    };
};



class Cubes {
public:
    std::vector<Cube> cubes;
    int               world_id;
    glm::u32vec2      world_size; 
    glm::vec3         position;
    glm::vec4         color;
    Cubes(int _world_id, glm::u32vec2 _world_size, glm::vec3 _pos);
    Cubes& addCube(Cube _o);
    Cubes& addObjects(std::vector<Cube> _os);
    Cubes& removeObject(Cube _o);
};
struct CompareVec3 {
    bool operator()(const glm::vec3& a, const glm::vec3& b) const {
        // 按分量逐个比较
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    }
};
class Scene {
public:
    std::vector<const Cube*> cubes;
    std::vector<const Cubes*> cubese;
    
    void addCube(const Cube* cube);
    void addCubes(const Cubes* Cubes);
    void removeCube(const Cube* cube);
    
    Scene& initScene();
    Scene() = default;
    ~Scene();
    std::map<const Cube*, size_t> cube_map;
    std::map<glm::vec3, size_t, CompareVec3>    pos_map;
};

enum class InstanceType {
    CUBE = 0
};

struct EnumHash {
    template <typename T>
    size_t operator()(T t) const
    {
        return static_cast<size_t>(t);
    }
};

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

    std::unordered_map<InstanceType, InstanceFirstLocation,EnumHash> instanceFirstLocations;

    BufferManager& init();
    void initCubeInstance();

    // 添加对象到缓冲区
    void addObject(const Object* obj);
    void addCube(const Cube* cube);
    void addCubes(const Cubes* Cubes);

    void removeCube(const Cube* cube);

    void addScene(Scene& scene);
    void updateObject(const Object* obj);
    void updateInstance(Cube& obj);

private:
    // 分配新的顶点块
    uint32_t allocateVertexBlock();

    // 分配新的索引块
    uint32_t allocateIndexBlock();

    uint32_t allocateInstanceBlock();

    // 上传数据到指定块的指定位置
    void uploadData(BufferBlock& block, const void* data, size_t size, uint32_t offset);

};


class Focus: public Object {

};

struct guiFlags {
    bool isStates;
};

class Role {
private:
    double lastLeftClickTime = 0.0;
    double lastRightClickTime = 0.0;
    const double clickCooldown = 0.2; // 0.1秒冷却时间

public:
    Role();
    ~Role() = default;
    const Cube* focus;
    Cube* place;
    Cube* current;
    Role& update();
    void Key();
    void MouseButton();
};