#include "gui.hpp"
#include "descriptor.hpp"
#include <stdexcept>
#include "camera.hpp"
#include <memory>
namespace gui {

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


    void draw_states() {
        ImGui::Begin("States");

        ImGui::Text("FPS: %.1f (%.2f ms/frame)", app.fps, app.frameTime);

        ImGui::Separator();
        ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera::pos.x, camera::pos.y, camera::pos.z);
        ImGui::Text("Camera Front: (%.2f, %.2f, %.2f)", camera::front.x, camera::front.y, camera::front.z);
        ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", camera::up.x, camera::up.y, camera::up.z);
        ImGui::Separator();

        if (app.role.focus != nullptr) {
            ImGui::Text("focus:(%.2f, %.2f, %.2f)", app.role.focus->pos.x, app.role.focus->pos.y, app.role.focus->pos.z);
        }
        if (app.role.place != nullptr) {
            ImGui::Text("place:(%.2f, %.2f, %.2f)", app.role.place->pos.x, app.role.place->pos.y, app.role.place->pos.z);
        }
        if (app.role.current != nullptr) {
            ImGui::Text("current:(%.2f, %.2f, %.2f)", app.role.current->pos.x, app.role.current->pos.y, app.role.current->pos.z);
        }
        int a = 0;
        for (auto& i : app.bufferM.instanceBlocks) {
            ImGui::Text("current instance block: id:(%d) , used:(%d) ,blocknum:(%d)", a, i.used, i.used / sizeof(InstanceData));
            a++;
        }
        ImGui::End();
    }

    void drawImGui() {
        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
        if (app.guiFlags.isStates) {
            draw_states();
        }

        ImGui::Render();

    }

    void cleanupImGui() {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

}