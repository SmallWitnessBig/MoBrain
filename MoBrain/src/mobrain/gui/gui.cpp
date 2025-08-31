#include "gui.hpp"
#include "camera/camera.hpp"
#include "neuronGui.hpp"
#include "synapseGui.hpp"
#include "usefulGui.hpp"
namespace gui {

    posMap<std::unordered_map<Name,NeuronGui*>> neuronGuis;
    posMap<std::unordered_map<Name,SynapseGui*>> synapseGuis;
    void initImGui() {

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
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
        auto& io=ImGui::GetIO();
        io.FontGlobalScale = 2.0f;
    }


    void draw_states() {
        ImGui::Begin("States",&app.guiFlags.isOpenStates);

        ImGui::Text("FPS: %.1f (%.2f ms/frame)", app.fps, app.frameTime);

        ImGui::Separator();
        ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", camera::pos.x, camera::pos.y, camera::pos.z);
        ImGui::Text("Camera Front: (%.2f, %.2f, %.2f)", camera::front.x, camera::front.y, camera::front.z);
        ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", camera::up.x, camera::up.y, camera::up.z);
        ImGui::Separator();

        if (app.role.focusCube != nullptr) {
            ImGui::Text("focus:(%.2f, %.2f, %.2f)", app.role.focusCube->pos.x, app.role.focusCube->pos.y, app.role.focusCube->pos.z);
        }
        if (app.role.place != nullptr) {
            ImGui::Text("current:(%.2f, %.2f, %.2f)", app.role.place->pos.x, app.role.place->pos.y, app.role.place->pos.z);
        }
        int a = 0;
        for (auto& i : app.bufferM.instanceBlocks) {
            ImGui::Text("current instance block: id:(%d) , used:(%d) ,blocknum:(%d)", a, i.used, i.used / sizeof(InstanceData));
            a++;
        }
        ImGui::End();
    }
    bool IsPresetNeuronGroup = false;


    void draw_settings() {
        ImGui::Begin("Settings",&app.guiFlags.isOpenNetGui);
        glm::vec3 pos = app.role.focusCube->pos;
        networkManager& n = app.net;
        int t = 0;
        Name toErase;
        for (auto& i : n.neuron_map[pos]) {
            ImGui::Text(i.first.c_str());
            ImGui::SameLine();
            right_align();
            ImGui::PushID(t);
            if (ImGui::Button("Gui")) {
                neuronGuis[pos][i.first] = new NeuronGui(pos,i.first);
            }
            ImGui::PopID();
            ImGui::SameLine();
            ImGui::PushID(t);
            if (ImGui::Button("Delete")) {
                toErase = i.first;
            }
            ImGui::PopID();
            t++;
        }
        if (!toErase.empty()) {
            if (neuronGuis[pos].contains(toErase)) {
                neuronGuis[pos].erase(toErase);
            }
            n.removeNeuronGroup(pos,toErase);
        }
        right_align();
        if (ImGui::Button("Set Preset")) {
            app.guiFlags.isOpenNeuronPreset = true;
        }
        ImGui::Checkbox("IsPresetNeuronGroup",&IsPresetNeuronGroup);
        ImGui::SameLine();
        right_align();
        if (ImGui::Button("Add Neuron Group")) {
            if (IsPresetNeuronGroup) {
                switch (app.neuronGroupPreset.type) {
                    case LIF: {
                        NeuronGroup* neuron_group = new LIFGroup(pos,app.neuronGroupPreset.num);

                        n.addNeuronGroup(neuron_group,"Default"+std::to_string(n.neuron_map[pos].size()));
                        break;
                    }
                    case HH: {
                        NeuronGroup* neuron_group = new HHGroup(pos,app.neuronGroupPreset.num);

                        n.addNeuronGroup(neuron_group,"Default"+std::to_string(n.neuron_map[pos].size()));
                        break;
                    }
                }
            }else {
                app.guiFlags.isOpenNeuronSettings = true;
            }
        }
        ImGui::End();
    }
    void draw_neuron_preset() {
        ImGui::Begin("Preset",&app.guiFlags.isOpenNeuronPreset);
        const char* items[] = { "LIF", "HH"};
        static int item_current = 0;
        ImGui::Combo("Neuron Type", &item_current, items, IM_ARRAYSIZE(items));
        switch (item_current) {
            case 0: {
                app.neuronGroupPreset.type = LIF;
                break;
            }
            case 1: {
                app.neuronGroupPreset.type = HH;
                break;
            }
            default: app.neuronGroupPreset.type = LIF;;
        }
        ImGui::SliderInt("num = ",&app.neuronGroupPreset.num,1,256);
        ImGui::SameLine();
        ImGui::Text("%d", app.neuronGroupPreset.num*128);
        ImGui::End();
    }
    char name[32]{};
    char null[32];
    int num = 1;
    void draw_neuron_settings() {
        ImGui::Begin("Neuron Settings",&app.guiFlags.isOpenNeuronSettings);
        ImGui::InputText("Name",name,32);
        ImGui::SliderInt("num = ",&num,1,16);
        ImGui::SameLine();
        ImGui::Text("%d", num*128);
        glm::vec3 pos = app.role.focusCube->pos;
        auto& n = app.net;
        if (ImGui::Button("Add")) {
            if (n.neuron_map[pos].contains(name)) {
                Error("Existed");
            }else if (strcmp(name,null)==0) {
                Error("Name can't be empty");
            }else{
                switch (app.neuronGroupPreset.type) {
                    case LIF: {
                        NeuronGroup* neuron_group = new LIFGroup(pos,app.neuronGroupPreset.num);
                        n.addNeuronGroup(neuron_group,name);
                        break;
                    }
                    case HH: {
                        NeuronGroup* neuron_group = new HHGroup(pos,app.neuronGroupPreset.num);
                        n.addNeuronGroup(neuron_group,name);
                        break;
                    }
                }
            }
        }
        ImGui::End();
    }
    void drawImGui() {
        ImGui_ImplGlfw_NewFrame();
        ImGui_ImplVulkan_NewFrame();
        ImGui::NewFrame();
        if (app.guiFlags.isOpenStates) {
            draw_states();
        }
        if (app.isFocus== true) {

            if (app.guiFlags.isOpenNetGui&&app.isFocus) {
                draw_settings();
            }
            if (app.guiFlags.isOpenNeuronPreset) {
                draw_neuron_preset();
            }
            if (app.guiFlags.isOpenNeuronSettings) {
                draw_neuron_settings();
            }
            for (auto& i : neuronGuis) {
                for (auto& j : i.second) {
                    j.second->draw();
                }
            }
            
            // 绘制突触连接
            
            if (toEraseNeuronGui) {
                if (neuronGuis[toEraseNeuronGui->pos].size()==1) {
                    neuronGuis.erase(toEraseNeuronGui->pos);
                }else {
                    neuronGuis[toEraseNeuronGui->pos].erase(toEraseNeuronGui->name);
                }
                delete toEraseNeuronGui;
                toEraseNeuronGui = nullptr;
            }
            for (auto& i : errorGuis) {
                i->draw();
            }
            if (toEraseErrorGui) {
                errorGuis.erase(std::ranges::find(errorGuis,toEraseErrorGui));
                delete toEraseErrorGui;
                toEraseErrorGui = nullptr;
            }
        }
        ImGui::Render();

    }

    void cleanupImGui() {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }

}