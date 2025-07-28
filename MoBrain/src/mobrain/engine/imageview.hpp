#pragma once
#include "vulkaninit.hpp"

void createImageViews();
vk::raii::ImageView createImageView(
    const vk::Image image,
    const vk::Format format,
    const vk::ImageAspectFlags aspectFlags
);