using System.IO;
using Newtonsoft.Json.Linq;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace SwarmAnima;

/// <summary>Native Anima reference and pose parameters.</summary>
public class SwarmAnimaExtension : Extension
{
    public static T2IRegisteredParam<Image> PoseImage;
    public static T2IRegisteredParam<double> PoseStrength, ReferenceStrength, ReferenceStart, ReferenceEnd;
    public static T2IRegisteredParam<int> PrimaryReferenceImage;

    public override void OnInit()
    {
        Description = "Reference images and pose control for Anima.";
        ExtensionAuthor = "tbbbtbbbtbbb";
        License = "Apache-2.0";
        Tags = ["parameters", "nodes"];
        ScriptFiles.Add("Assets/anima.js");
        ComfyUISelfStartBackend.CustomNodePaths.Add(Path.GetFullPath($"{FilePath}ExtraNodes"));
        ComfyUIBackendExtension.NodeToFeatureMap["SwarmAnimaReference"] = "swarmanima";

        T2IParamGroup group = new("Anima", Toggles: false, Open: false);
        PoseImage = T2IParamTypes.Register<Image>(new("Pose Image",
            "Detect and reuse the pose of the first person in this image. For Anima models.",
            null, Group: group, FeatureFlag: "swarmanima", Toggleable: true, OrderPriority: 1, ImageShouldResize: false));
        PoseStrength = T2IParamTypes.Register<double>(new("Pose Strength",
            "How strongly the generated image follows Pose Image. Zero disables pose control.",
            "0.8", Min: 0, Max: 2, Step: 0.05, Group: group, FeatureFlag: "swarmanima", OrderPriority: 2,
            ViewType: ParamViewType.SLIDER, DependNonDefault: PoseImage.Type.ID));
        ReferenceStrength = T2IParamTypes.Register<double>(new("Reference Strength",
            "Use one image from Swarm's Image Prompt input to guide Anima's appearance. Zero disables it. Extra images are ignored.",
            "1", Min: 0, Max: 2, Step: 0.05, Group: group, FeatureFlag: "swarmanima", OrderPriority: 3,
            ViewType: ParamViewType.SLIDER));
        PrimaryReferenceImage = T2IParamTypes.Register<int>(new("Primary Reference Image",
            "Which Image Prompt image to use, counting from 1. Only this image conditions Anima.",
            "1", Min: 1, Max: 100, Group: group, FeatureFlag: "swarmanima", OrderPriority: 4, IsAdvanced: true));
        ReferenceStart = T2IParamTypes.Register<double>(new("Reference Start",
            "Start reference conditioning at this fraction of denoising (0 = beginning).",
            "0", Min: 0, Max: 1, Step: 0.05, Group: group, FeatureFlag: "swarmanima", OrderPriority: 5, IsAdvanced: true));
        ReferenceEnd = T2IParamTypes.Register<double>(new("Reference End",
            "End reference conditioning at this fraction of denoising (1 = end).",
            "1", Min: 0, Max: 1, Step: 0.05, Group: group, FeatureFlag: "swarmanima", OrderPriority: 6, IsAdvanced: true));

        WorkflowGenerator.AddStep(Apply, -6.5);
    }

    private static void Apply(WorkflowGenerator g)
    {
        if (!g.IsAnima())
        {
            return;
        }
        bool hasReference = g.UserInput.TryGet(T2IParamTypes.PromptImages, out List<Image> images) && images.Count > 0;
        bool hasPose = g.UserInput.TryGet(PoseImage, out Image pose);
        double referenceStrength = g.UserInput.Get(ReferenceStrength, 1);
        double poseStrength = g.UserInput.Get(PoseStrength, 0.8);
        if ((!hasReference || referenceStrength == 0) && (!hasPose || poseStrength == 0))
        {
            return;
        }
        if (!g.Features.Contains("swarmanima"))
        {
            throw new SwarmUserErrorException("Install SwarmAnima's ExtraNodes on this ComfyUI backend and restart it.");
        }
        if (hasPose && poseStrength > 0)
        {
            WGNodeData image = g.LoadImage(pose, "${poseimage}", false);
            string detected = g.CreateNode("SwarmAnimaPose", new JObject
            {
                ["image"] = image.Path,
                ["width"] = g.UserInput.Get(T2IParamTypes.Width),
                ["height"] = g.UserInput.Get(T2IParamTypes.Height)
            });
            string encoded = g.CreateNode("VAEEncode", new JObject
            {
                ["pixels"] = new JArray(detected, 0), ["vae"] = g.CurrentVae.Path
            });
            string applied = g.CreateNode("SwarmAnimaPoseApply", new JObject
            {
                ["model"] = g.CurrentModel.Path, ["control_latent"] = new JArray(encoded, 0), ["strength"] = poseStrength
            });
            g.CurrentModel = g.CurrentModel.WithPath([applied, 0]);
        }
        if (hasReference && referenceStrength > 0)
        {
            int index = g.UserInput.Get(PrimaryReferenceImage, 1) - 1;
            if (index < 0 || index >= images.Count)
            {
                throw new SwarmUserErrorException($"Primary Reference Image must be between 1 and {images.Count}.");
            }
            double start = g.UserInput.Get(ReferenceStart, 0), end = g.UserInput.Get(ReferenceEnd, 1);
            if (start >= end)
            {
                throw new SwarmUserErrorException("Reference Start must be less than Reference End.");
            }
            WGNodeData image = g.LoadImage(images[index], "${promptimages." + index + "}", false);
            string encoded = g.CreateNode("SwarmAnimaEncode", new JObject { ["image"] = image.Path });
            string applied = g.CreateNode("SwarmAnimaReference", new JObject
            {
                ["model"] = g.CurrentModel.Path, ["features"] = new JArray(encoded, 0),
                ["strength"] = referenceStrength, ["start"] = start, ["end"] = end
            });
            g.CurrentModel = g.CurrentModel.WithPath([applied, 0]);
        }
    }
}
