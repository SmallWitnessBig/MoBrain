#pragma once

#include "context.hpp"

void createImage(
    const uint32_t width,
    const uint32_t height,
    const vk::Format format,
    const vk::ImageTiling tiling,
    const vk::ImageUsageFlags usage,
    const vk::MemoryPropertyFlags properties,
    vk::raii::Image& image,
    vk::raii::DeviceMemory& imageMemory
);