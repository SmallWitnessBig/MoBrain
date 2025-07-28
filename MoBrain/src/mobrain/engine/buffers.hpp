#pragma once
#include "vulkaninit.hpp"
vk::Format findDepthFormat( const std::vector<vk::Format>& candidates );
uint32_t findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties);
// 创建帧缓冲区
void createFramebuffers();
// 创建顶点缓冲区
void createVertexBuffer(
    std::vector<Vertex>& _vertices,
    vk::raii::Buffer& _buffer,
    vk::raii::DeviceMemory& _bufferMemory
);
// 创建索引缓冲区
void createIndexBuffer(
    std::vector<uint32_t>& _indices,
    vk::raii::Buffer& _buffer,
    vk::raii::DeviceMemory& _bufferMemory
);
void createUniformBuffers();
void createDepthResources();
// 创建缓冲区
// @param size: 缓冲区大小
// @param usage: 缓冲区使用标志
// @param properties: 内存属性标志
// @param bufferMemory: 输出参数，用于存储分配的内存
// @param buffer: 输出参数，用于存储创建的缓冲区
void createBuffer(
    const vk::DeviceSize size,
    const vk::BufferUsageFlags usage,
    const vk::MemoryPropertyFlags properties,
    vk::raii::DeviceMemory& bufferMemory,
    vk::raii::Buffer& buffer
);
// 复制缓冲区内容
// @param buffersource: 源缓冲区
// @param bufferdst: 目标缓冲区
// @param size: 要复制的数据大小
void copyBuffer(
    const vk::raii::Buffer& buffersource,
    const vk::raii::Buffer& bufferdst,
    const vk::DeviceSize size,
    const vk::DeviceSize offset = 0
);
