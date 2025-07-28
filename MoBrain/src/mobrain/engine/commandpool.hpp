#pragma once
#include "vulkaninit.hpp"

void createFramebuffers();
void createCommandPool();
void createCommandBuffers();
void recordCommandBuffer(const vk::raii::CommandBuffer& commandbuffer, uint32_t image_index);
