//
// Created by 31530 on 2025/8/13.
//

#ifndef MOBRAIN_GRAPHICS_HPP
#define MOBRAIN_GRAPHICS_HPP

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vulkan/vulkan.hpp>

#define MIN_OBJECT_VERTICES_COUNT 8
#define MIN_OBJECT_INDICES_COUNT 36
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

enum class InstanceType {
    CUBE = 0
};

//Cube

class Cube : public InstanceData {
public:
    glm::vec3 pos;
    Cube(glm::vec3 pos, glm::vec4 color) {
        this->pos = pos;
        this->color = color;
        model = glm::translate(glm::mat4{ 1.0f }, pos);
    };
    Cube(const Cube& c) {
        color = c.color;
        model = c.model;
        pos = c.pos;
    };
};
#endif //MOBRAIN_GRAPHICS_HPP