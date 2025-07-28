#include "gui.hpp"
#include "descriptor.hpp"
#include <stdexcept>
#include "camera.hpp"
#include <memory>
void initImGui() {


    // Initialize ImGui
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    //ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(app.window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};

	init_info.ApiVersion = VK_API_VERSION_1_0;
    init_info.Instance = *app.instance;
    init_info.PhysicalDevice = *app.physicalDevice;
    init_info.Device = *app.device;
    init_info.QueueFamily = 0;
    init_info.Queue = *app.graphicsQueue;
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = *app.descriptorPool;
    init_info.Allocator = nullptr;
    init_info.MinImageCount = 2;
    init_info.ImageCount = app.swapChainImages.size();
    init_info.CheckVkResultFn = nullptr;
    init_info.RenderPass = *app.renderPass;
    ImGui_ImplVulkan_Init(&init_info);
    

}

static int x = 0;
static int y = 0;
static int z = 0;
static ImColor color = ImColor(1.0f, 1.0f, 1.0f, 1.0f); // 默认白色

void drawImGui() {
    ImGui_ImplGlfw_NewFrame();
    ImGui_ImplVulkan_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Vulkan Cube Placer");

    ImGui::Text("FPS: %.1f (%.2f ms/frame)", app.fps, app.frameTime);

    ImGui::Separator();
    ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera::pos.x, camera::pos.y, camera::pos.z);
    ImGui::Text("Camera Front: (%.2f, %.2f, %.2f)", camera::front.x, camera::front.y, camera::front.z);
    ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", camera::up.x, camera::up.y, camera::up.z);
    ImGui::Separator();
    ImGui::SliderInt("x", &x, -10, 10);
    ImGui::SliderInt("y", &y, -10, 10);
    ImGui::SliderInt("z", &z, -10, 10);
    ImGui::ColorEdit4("Color", &color.Value.x);
    ImGui::SameLine();
    ImGui::ColorButton("Color", color);
    if (ImGui::Button("new Cube")) {
        auto cube = std::make_shared<Cube>( glm::vec3{x,y,z} ,glm::vec3{color.Value.x,color.Value.y,color.Value.z} );
        std::cout << color.Value.x << color.Value.y << color.Value.z;
        app.main_scene.addObject(cube);
        app.render_scene.addObject(cube);
    }
    ImGui::End();
    ImGui::Render();
    
}

void cleanupImGui() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(); 
}
