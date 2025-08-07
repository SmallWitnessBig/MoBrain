#include "buffers.hpp"
#include <queue>
#include "imageview.hpp"
#include "image.hpp"
/**
 * @brief 创建Vulkan缓冲区
 *
 * 创建具有指定大小和用途的Vulkan缓冲区，分配符合指定属性的内存，
 * 并将内存绑定到缓冲区。
 *
 * @param size 缓冲区大小（字节）
 * @param usage 缓冲区用途标志，指定其用途
 * @param properties 内存属性标志，指定所需的内存属性
 * @param bufferMemory 用于存储分配的内存对象的引用
 * @param buffer 用于存储创建的缓冲区对象的引用
 */
void createBuffer(
    const vk::DeviceSize size,
    const vk::BufferUsageFlags usage,
    const vk::MemoryPropertyFlags properties,
    vk::raii::DeviceMemory& bufferMemory,
    vk::raii::Buffer& buffer
) {
    vk::BufferCreateInfo bufferCreate;
    bufferCreate
        .setSize(size)
        .setUsage(usage)
        .setSharingMode(vk::SharingMode::eExclusive);
    buffer = app.device.createBuffer(bufferCreate);

    const auto mem = buffer.getMemoryRequirements();

    vk::MemoryAllocateInfo allocInfo;
    allocInfo
        .setAllocationSize(mem.size)
        .setMemoryTypeIndex(findMemoryType(mem.memoryTypeBits, properties));

    try {
        bufferMemory = app.device.allocateMemory(allocInfo);
        buffer.bindMemory(*bufferMemory, 0);
    }
    catch (const vk::SystemError& err) {
        if (err.code() == vk::Result::eErrorOutOfDeviceMemory) {
            throw std::runtime_error("Out of device memory. Consider reducing buffer sizes or implementing memory pooling.");
        }
        else {
            throw;
        }
    }
}
void coverBuffer(
    const vk::raii::Buffer& buffersrc,
    const vk::DeviceSize srcOffset,
    const vk::raii::Buffer& bufferdst,
    const vk::DeviceSize dstOffset,
    const vk::DeviceSize size
    ) {
    vk::CommandBufferAllocateInfo allocInfo;
    allocInfo
        .setCommandPool(app.commandPool)
        .setCommandBufferCount(1)
        .setLevel(vk::CommandBufferLevel::ePrimary); // 主命令缓冲区
    auto commandBuffers = app.device.allocateCommandBuffers(allocInfo);

    const auto commandBuffer = std::move(commandBuffers.at(0));

    vk::CommandBufferBeginInfo beginInfo;
    beginInfo.setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit); // 命令缓冲区只能提交一次
    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion;
    copyRegion
        .setSrcOffset(srcOffset)
        .setDstOffset(dstOffset)
        .setSize(size);
    commandBuffer.copyBuffer(*buffersrc, *bufferdst, copyRegion);

    commandBuffer.end();
    // 立即执行内存传输命令
    vk::SubmitInfo submitInfo;
    submitInfo
        .setCommandBufferCount(1)
        .setCommandBuffers(*commandBuffer);
    vk::raii::Fence fence{ app.device, vk::FenceCreateInfo() };
    app.graphicsQueue.submit(submitInfo, *fence);
    constexpr uint64_t TIMEOUT = 1'000'000'000;
    vk::Result result = app.device.waitForFences({ *fence }, VK_TRUE, TIMEOUT);

    if (result != vk::Result::eSuccess) {
        throw std::runtime_error("Buffer copy operation timed out");
    }

}
/**
 * @brief 复制Vulkan缓冲区数据
 *
 * 从一个Vulkan缓冲区复制指定大小的数据到另一个缓冲区。
 * 该函数创建主命令缓冲区，记录缓冲区复制操作，并使用一次性提交标志提交，
 * 以确保命令只执行一次。最后提交命令到图形队列并等待操作完成。
 *
 * @param buffersource 包含要复制数据的源缓冲区
 * @param bufferdst 接收复制数据的目标缓冲区
 * @param size 要复制的数据大小（字节）
 */
void copyBuffer(
    const vk::raii::Buffer& buffersrc,
    const vk::raii::Buffer& bufferdst,
    const vk::DeviceSize size,
    const vk::DeviceSize offset
) {

    coverBuffer(
        buffersrc,
        0,
        bufferdst,
        offset,
        size
    );

}




/**
 * @brief 查找符合要求的内存类型
 * 
 * 根据给定的内存类型过滤器和内存属性要求，查找最适合的内存类型索引。
 * 
 * @param typeFilter 内存类型过滤器，指定可用的内存类型
 * @param properties 所需的内存属性标志
 * @return 找到的内存类型索引
 */
uint32_t findMemoryType(const uint32_t typeFilter, const vk::MemoryPropertyFlags properties) {
    const auto memproperties = app.physicalDevice.getMemoryProperties();
    for (uint32_t i = 0; i < memproperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memproperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}
/**
 * @brief 创建交换链帧缓冲区
 *
 * 为每个交换链图像视图创建对应的帧缓冲区，用于渲染操作。
 * 帧缓冲区将图像视图与渲染过程关联起来。
 */
void createFramebuffers() {
    app.swapChainFramebuffers.reserve(app.swapChainImageViews.size());

    vk::FramebufferCreateInfo framebufferInfo;
    framebufferInfo.renderPass = app.renderPass;
    framebufferInfo.width = app.swapChainExtent.width;
    framebufferInfo.height = app.swapChainExtent.height;
    framebufferInfo.layers = 1;
    for (const auto& imageView : app.swapChainImageViews) {
        const std::array<vk::ImageView, 2> imageViews{ imageView, app.depthImageView };
        framebufferInfo.setAttachments(imageViews);
        app.swapChainFramebuffers.emplace_back(app.device.createFramebuffer(framebufferInfo));
    }
}
/**
 * @brief 创建顶点缓冲区
 * 
 * 创建用于存储顶点数据的缓冲区。首先创建一个主机可见的临时缓冲区来上传数据，
 * 然后创建设备本地的最终缓冲区，并将数据从临时缓冲区复制到最终缓冲区。
 */
void createVertexBuffer(
    std::vector<Vertex>& _vertices,
    vk::raii::Buffer& _buffer,
    vk::raii::DeviceMemory& _bufferMemory
) {
    const vk::DeviceSize bufferSize = sizeof(Vertex) * _vertices.size();

    // 创建用于顶点数据传输的临时缓冲区
    vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
    vk::raii::Buffer stagingBuffer{ nullptr };
    createBuffer(bufferSize, 
        vk::BufferUsageFlagBits::eTransferSrc,  // 内存传输的源
        vk::MemoryPropertyFlagBits::eHostVisible | 
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBufferMemory, 
        stagingBuffer
    );

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, _vertices.data(), static_cast<size_t>(bufferSize));
    stagingBufferMemory.unmapMemory();

    // 在设备本地内存中创建最终的顶点缓冲区
    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst |  // 内存传输的目标
        vk::BufferUsageFlagBits::eVertexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        _bufferMemory,
        _buffer
    );

    copyBuffer(stagingBuffer, _buffer, bufferSize);
}

/**
 * @brief 创建索引缓冲区
 * 
 * 创建用于存储索引数据的缓冲区。首先创建一个主机可见的临时缓冲区来上传数据，
 * 然后创建设备本地的最终缓冲区，并将数据从临时缓冲区复制到最终缓冲区。
 */
void createIndexBuffer(
    std::vector<uint32_t>& _indices,
    vk::raii::Buffer& _buffer,
    vk::raii::DeviceMemory& _bufferMemory
) {
    const vk::DeviceSize bufferSize = sizeof(uint32_t) * _indices.size();

    // 创建用于索引数据传输的临时缓冲区
    vk::raii::DeviceMemory stagingBufferMemory{ nullptr };
    vk::raii::Buffer stagingBuffer{ nullptr };
    createBuffer(bufferSize, 
        vk::BufferUsageFlagBits::eTransferSrc,  // 内存传输的源
        vk::MemoryPropertyFlagBits::eHostVisible |
        vk::MemoryPropertyFlagBits::eHostCoherent,
        stagingBufferMemory, 
        stagingBuffer
    );

    void* data = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(data, _indices.data(), static_cast<size_t>(bufferSize));
    stagingBufferMemory.unmapMemory();

    // 在设备本地内存中创建最终的索引缓冲区
    createBuffer(
        bufferSize,
        vk::BufferUsageFlagBits::eTransferDst |  
        vk::BufferUsageFlagBits::eIndexBuffer,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        _bufferMemory,
        _buffer
    );

    copyBuffer(stagingBuffer, _buffer, bufferSize);
}

/**
 * @brief 创建统一缓冲区
 * 
 * 为每一帧创建统一缓冲区，用于存储变换矩阵等每帧变化的数据。
 * 这些缓冲区是主机可见的，以便每帧更新数据。
 */
void createUniformBuffers() {
    const vk::DeviceSize bufferSize = sizeof(glm::mat4);

    for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        // 创建缓冲区内存对象
        vk::raii::DeviceMemory uniformBufferMemory{nullptr};
        vk::raii::Buffer uniformBuffer{nullptr};
        
        try {
            createBuffer(bufferSize,
                vk::BufferUsageFlagBits::eUniformBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible |
                vk::MemoryPropertyFlagBits::eHostCoherent,
                uniformBufferMemory,
                uniformBuffer
            );

            // 映射内存以便更新
            void* mappedMemory = uniformBufferMemory.mapMemory(0, bufferSize);
            
            // 存储到应用状态中
            app.uniformBuffersMemory.push_back(std::move(uniformBufferMemory));
            app.uniformBuffers.push_back(std::move(uniformBuffer));
            app.uniformBuffersMapped.push_back(mappedMemory);
        }
        catch (const vk::SystemError& err) {
            if (err.code() == vk::Result::eErrorOutOfDeviceMemory) {
                throw std::runtime_error("Failed to allocate uniform buffer memory: Out of device memory");
            }
            else {
                throw;
            }
        }
    }
}

//深度资源

vk::Format findDepthFormat(const std::vector<vk::Format>& candidates) {
    for (const vk::Format format : candidates) {
        const auto props = app.physicalDevice.getFormatProperties(format);
        if (props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
            return format;
        }
    }
    throw std::runtime_error("failed to find supported format!");
}

void createDepthResources() {
    const vk::Format depthFormat = findDepthFormat({ vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint });
    createImage(
        app.swapChainExtent.width,
        app.swapChainExtent.height,
        depthFormat,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eDepthStencilAttachment,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        app.depthImage,
        app.depthImageMemory
    );
    app.depthImageView = createImageView(app.depthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);

};


//缓冲分配
BufferManager& BufferManager::init() {
    // 预分配初始块
    allocateVertexBlock();
    allocateIndexBlock();
    allocateInstanceBlock();

    // 初始化Cube实例
    initCubeInstance();
    return *this;
}

void BufferManager::initCubeInstance() {
    const size_t vertexSize = sizeof(Vertex) * 8;
    const size_t indexSize  = sizeof(uint32_t) * 36;

    // 查找合适的顶点块
    uint32_t vertexBlockIndex = UINT32_MAX;
    for (auto& iter : freeVertexBlockIndices) {
        if (vertexSize <= vertexBlocks[iter].capacity - vertexBlocks[iter].used) {
            vertexBlockIndex = iter;
            break;
        }
    }
    if (vertexBlockIndex == UINT32_MAX) {
        vertexBlockIndex = allocateVertexBlock();
    }
    // 查找合适的块
    uint32_t indexBlockIndex = UINT32_MAX;
    for (auto& iter : freeIndexBlockIndices) {
        if (indexSize <= indexBlocks[iter].capacity - indexBlocks[iter].used) {
            indexBlockIndex = iter;
            break;
        }
    }
    if (indexBlockIndex == UINT32_MAX) {
        indexBlockIndex = allocateIndexBlock();
    }

    // 记录对象位置
    InstanceFirstLocation loc;
    loc.vertexBlockIndex = vertexBlockIndex;
    loc.indexBlockIndex = indexBlockIndex;
    loc.vertexOffset = static_cast<uint32_t>(vertexBlocks[vertexBlockIndex].used);
    loc.indexOffset = static_cast<uint32_t>(indexBlocks[indexBlockIndex].used);

    instanceFirstLocations[InstanceType::CUBE] = loc;

    std::vector<uint32_t> indices = {
        2, 1, 0, 0, 3, 2,
        4, 5, 6, 6, 7, 4,
        2, 3, 7, 7, 6, 2,
        0, 1, 5, 5, 4, 0,
        1, 2, 6, 6, 5, 1,
        0, 4, 7, 7, 3, 0
    };

    std::vector<Vertex> vertices = {
        {glm::vec3(-0.5f, -0.5f,-0.5f)},
        {glm::vec3(0.5f, -0.5f,-0.5f)},
        {glm::vec3(0.5f, 0.5f,-0.5f)},
        {glm::vec3(-0.5f, 0.5f,-0.5f)},
        {glm::vec3(-0.5f, -0.5f,0.5f)},
        {glm::vec3(0.5f, -0.5f,0.5f)},
        {glm::vec3(0.5f, 0.5f,0.5f)},
        {glm::vec3(-0.5f, 0.5f,0.5f)}
    };

    uploadData(vertexBlocks[vertexBlockIndex],
        vertices.data(),
        vertexSize,
        loc.vertexOffset);

    uploadData(indexBlocks[indexBlockIndex],
        indices.data(),
        indexSize,
        loc.indexOffset);

    vertexBlocks[vertexBlockIndex].used += vertexSize;

    if (vertexBlocks[vertexBlockIndex].capacity - vertexBlocks[vertexBlockIndex].used < MIN_OBJECT_VERTICES_COUNT * sizeof(Vertex)) {
        freeVertexBlockIndices.erase(std::remove(freeVertexBlockIndices.begin(), freeVertexBlockIndices.end(), vertexBlockIndex));
        std::cout << "block" << std::endl;

    }

    indexBlocks[indexBlockIndex].used += indexSize;
    if (indexBlocks[indexBlockIndex].capacity - indexBlocks[indexBlockIndex].used < MIN_OBJECT_INDICES_COUNT * sizeof(uint32_t)) {
        freeVertexBlockIndices.erase(std::remove(freeVertexBlockIndices.begin(), freeVertexBlockIndices.end(), vertexBlockIndex));
    }
}


// 添加对象到缓冲区
void BufferManager::addObject(const Object* obj) {
    const size_t vertexSize = obj->vertices.size() * sizeof(Vertex);
    const size_t indexSize = obj->indices.size() * sizeof(uint32_t);
    // 查找合适的顶点块
    uint32_t vertexBlockIndex = UINT32_MAX;
    for (auto& iter : freeVertexBlockIndices) {
        if (vertexSize <= vertexBlocks[iter].capacity - vertexBlocks[iter].used) {
            vertexBlockIndex = iter;
            break;
        }
    }
    if (vertexBlockIndex == UINT32_MAX) {
        vertexBlockIndex = allocateVertexBlock();
    }
    // 查找合适的块
    uint32_t indexBlockIndex = UINT32_MAX;
    for (auto& iter : freeIndexBlockIndices) {
        if (indexSize <= indexBlocks[iter].capacity - indexBlocks[iter].used) {
            indexBlockIndex = iter;
            break;
        }
    }
    if (indexBlockIndex == UINT32_MAX) {
        indexBlockIndex = allocateIndexBlock();
    }

    // 记录对象位置
    ObjectLocation loc;
    loc.vertexBlockIndex = vertexBlockIndex;
    loc.indexBlockIndex = indexBlockIndex;
    loc.vertexOffset = static_cast<uint32_t>(vertexBlocks[vertexBlockIndex].used);
    loc.indexOffset = static_cast<uint32_t>(indexBlocks[indexBlockIndex].used);
    objectLocations[obj] = loc;
    // 1 - > 0 0 0 0
    // 2 - > 0 0 
    // 上传数据
    uploadData(vertexBlocks[vertexBlockIndex],
        obj->vertices.data(),
        vertexSize,
        loc.vertexOffset);

    uploadData(indexBlocks[indexBlockIndex],
        obj->indices.data(),
        indexSize,
        loc.indexOffset);

    // 更新使用量
    vertexBlocks[vertexBlockIndex].used += vertexSize;
    if (vertexBlocks[vertexBlockIndex].capacity - vertexBlocks[vertexBlockIndex].used < MIN_OBJECT_VERTICES_COUNT * sizeof(Vertex)) {
        freeVertexBlockIndices.erase(std::remove(freeVertexBlockIndices.begin(), freeVertexBlockIndices.end(), vertexBlockIndex));
        std::cout << "block" << std::endl;

    }


    indexBlocks[indexBlockIndex].used += indexSize;
    if (indexBlocks[indexBlockIndex].capacity - indexBlocks[indexBlockIndex].used < MIN_OBJECT_INDICES_COUNT * sizeof(uint32_t)) {
        freeVertexBlockIndices.erase(std::remove(freeVertexBlockIndices.begin(), freeVertexBlockIndices.end(), vertexBlockIndex));
    }
}

void BufferManager::addCube(const Cube* cube) {
    const size_t cubeSize = sizeof(InstanceData);
    
    uint32_t instanceBlockIndex = UINT32_MAX;

    for (auto& iter : freeInstanceBlockIndices) {
        if (instanceBlocks[iter].capacity - instanceBlocks[iter].used >= cubeSize) {
            instanceBlockIndex = iter;
            break;
        }
    }
    if (instanceBlockIndex == UINT32_MAX) {
        instanceBlockIndex = allocateInstanceBlock();
    }

    InstanceLocation loc;
    loc.instanceBlockIndex = instanceBlockIndex;
    loc.instanceOffset = instanceBlocks[instanceBlockIndex].used;

    InstanceData _cube{ cube->model,cube->color };

    uploadData(
        instanceBlocks[instanceBlockIndex],
        &_cube,
        cubeSize,
        loc.instanceOffset
    );
    instanceLocations[cube] = loc;
    instanceBlocks[instanceBlockIndex].used += cubeSize;
    if (instanceBlocks[instanceBlockIndex].capacity -instanceBlocks[instanceBlockIndex].used< sizeof(InstanceData)){
        freeInstanceBlockIndices.erase(std::remove(freeInstanceBlockIndices.begin(), freeInstanceBlockIndices.end(), instanceBlockIndex));
    }
};

void BufferManager::addCubes(const Cubes* Cubes) {
    if (Cubes!=nullptr)
    {
        auto cube = new Cube(Cubes->position, Cubes->color);
        addCube(cube);
    }
}


// 删除对象
// 1. 从对象列表中删除对象
// 2. 把最后数据搬运到这个位置
// 3. 更新索引数据
void BufferManager::removeCube(const Cube* cube) {
    // 获取被移除对象信息
    auto loc = instanceLocations[cube];
    auto& block = instanceBlocks[loc.instanceBlockIndex];

    // 判断是否是最后一个实例
    if (loc.instanceOffset != block.used - sizeof(InstanceData)) {
        // 移动最后一个实例的数据到被删除位置
        coverBuffer(
            block.buffer, block.used - sizeof(InstanceData),
            block.buffer, loc.instanceOffset,
            sizeof(InstanceData)
        );

        // === 新增：更新被移动实例的位置信息 ===
        vk::DeviceSize lastOffset = block.used - sizeof(InstanceData);
        for (auto& [otherCube, otherLoc] : instanceLocations) {
            if (otherLoc.instanceBlockIndex == loc.instanceBlockIndex &&
                otherLoc.instanceOffset == lastOffset) {
                // 更新位置为被删除实例的偏移量
                otherLoc.instanceOffset = loc.instanceOffset;
                break; // 唯一匹配，直接退出
            }
        }
    }

    block.used -= sizeof(InstanceData);
    instanceLocations.erase(cube);

    // 标记空闲块（逻辑不变）
    bool is_emplaced = false;
    for (auto& i : freeInstanceBlockIndices) {
        if (i == loc.instanceBlockIndex) {
            is_emplaced = true;
            break;
        }
    }
    if (!is_emplaced) {
        freeInstanceBlockIndices.emplace_back(loc.instanceBlockIndex);
    }
}
// 更新对象数据
void BufferManager::updateObject(const Object* obj) {
    const auto& loc = objectLocations.at(obj);
    const size_t vertexSize = obj->vertices.size() * sizeof(Vertex);
    const size_t indexSize = obj->indices.size() * sizeof(uint32_t);

    uploadData(vertexBlocks[loc.vertexBlockIndex],
        obj->vertices.data(),
        vertexSize,
        loc.vertexOffset);

    uploadData(indexBlocks[loc.indexBlockIndex],
        obj->indices.data(),
        indexSize,
        loc.indexOffset);
}

// 分配新的顶点块
uint32_t BufferManager::allocateVertexBlock() {
    BufferBlock block;
    vk::DeviceSize blockSize = VERTEX_BLOCK_SIZE;
    
    try {
        createBuffer(blockSize,
            vk::BufferUsageFlagBits::eVertexBuffer |
            vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal,
            block.memory,
            block.buffer);
    }
    catch (const std::runtime_error& err) {
        // 如果分配失败，尝试更小的块大小
        blockSize = VERTEX_BLOCK_SIZE / 2;
        while (blockSize >= 1024 * 1024) { // 最小1MB
            try {
                createBuffer(blockSize,
                    vk::BufferUsageFlagBits::eVertexBuffer |
                    vk::BufferUsageFlagBits::eTransferDst,
                    vk::MemoryPropertyFlagBits::eDeviceLocal,
                    block.memory,
                    block.buffer);
                break; // 成功分配，跳出循环
            }
            catch (...) {
                blockSize /= 2; // 继续减小块大小
            }
        }
        
    }
    
    block.capacity = blockSize;
    block.used = 0;

    vertexBlocks.push_back(std::move(block));
    uint32_t index = static_cast<uint32_t>(vertexBlocks.size()) - 1;
    freeVertexBlockIndices.push_back(index);

    return index;
}

// 分配新的索引块
uint32_t BufferManager::allocateIndexBlock() {
    BufferBlock block;
    vk::DeviceSize blockSize = INDEX_BLOCK_SIZE;
    
    while (blockSize >= 512 * 1024) {
        try {
            createBuffer(blockSize,
                vk::BufferUsageFlagBits::eIndexBuffer |
                vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                block.memory,
                block.buffer);
            break; // 成功分配，跳出循环
        }
        catch (...) {
            blockSize /= 2; // 继续减小块大小
        }
    }
    
    block.capacity = blockSize;
    block.used = 0;

    indexBlocks.push_back(std::move(block));

    uint32_t index = static_cast<uint32_t>(indexBlocks.size()) - 1;
    freeIndexBlockIndices.push_back(index);

    return index;
}
// 分配新的索引块
uint32_t BufferManager::allocateInstanceBlock() {
    BufferBlock block;
    vk::DeviceSize blockSize = INSTANCE_BLOCK_SIZE;

    while (blockSize >= 512 * 1024) {
        try {
            createBuffer(blockSize,
                vk::BufferUsageFlagBits::eVertexBuffer | 
                vk::BufferUsageFlagBits::eTransferSrc |
                vk::BufferUsageFlagBits::eTransferDst,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                block.memory,
                block.buffer);
            break; // 成功分配，跳出循环
        }
        catch (...) {
            blockSize /= 2; // 继续减小块大小
        }
    }

    block.capacity = blockSize;
    block.used = 0;

    instanceBlocks.push_back(std::move(block));

    uint32_t index = static_cast<uint32_t>(instanceBlocks.size()) - 1;
    freeInstanceBlockIndices.push_back(index);

    return index;
}

// 上传数据到指定块的指定位置
void BufferManager::uploadData(BufferBlock& block, const void* data, size_t size, uint32_t offset) {
    // 创建临时暂存缓冲区
    vk::raii::DeviceMemory stagingMemory{ nullptr };
    vk::raii::Buffer stagingBuffer{ nullptr };
    try{
        createBuffer(size,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible |
            vk::MemoryPropertyFlagBits::eHostCoherent,
            stagingMemory,
            stagingBuffer);
    }
    catch(std::runtime_error& e){
        std::cout << "Failed to create staging buffer" << std::endl;
    }


    // 填充暂存缓冲区
    void* stagingData = stagingMemory.mapMemory(0, size);
    memcpy(stagingData, data, size);
    stagingMemory.unmapMemory();

    // 创建缓冲区
    copyBuffer(stagingBuffer, block.buffer, size, offset);
}
