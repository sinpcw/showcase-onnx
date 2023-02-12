using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Configuration;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


static void PrintConfigure(string mode)
{
    Console.WriteLine("{0} Inference", mode);
    Console.WriteLine("  rootdir : {0}", ReadParameterAsStr("rootdir", "./"));
    Console.WriteLine("  datadir : {0}", ReadParameterAsStr("datadir", "./"));
    Console.WriteLine("  imgsize : {0}", ReadParameterAsInt("imgsize", 0));
}

static string ReadParameterAsStr(string key, string defaultValue = "")
{
    var s = ConfigurationManager.AppSettings[key];
    if (s == null)
    {
        return defaultValue;
    }
    return s;
}

static int ReadParameterAsInt(string key, int defaultValue = 0)
{
    var s = ConfigurationManager.AppSettings[key];
    if (s == null)
    {
        return defaultValue;
    }
    return int.Parse(s);
}
static Bitmap ImageResize(Image image, int width, int height)
{
    var rect = new Rectangle(0, 0, width, height);
    var dest = new Bitmap(width, height);
    dest.SetResolution(image.HorizontalResolution, image.VerticalResolution);
    using (var graphics = Graphics.FromImage(dest))
    {
        graphics.InterpolationMode = InterpolationMode.Bilinear;
        graphics.DrawImage(image, rect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel);
    }
    return dest;
}


static Tensor<float> GetImageTensor(string filepath)
{
    var imageSize = ReadParameterAsInt("imgsize");
    var tensor = new DenseTensor<float>(new[] { 1, 3, imageSize, imageSize });
    using (var bitmap = new Bitmap(filepath))
    using (var buffer = ImageResize(bitmap, imageSize, imageSize))
    {
        var lockbits = buffer.LockBits(new Rectangle(0, 0, buffer.Width, buffer.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var stream = new byte[lockbits.Stride * buffer.Height];
            System.Runtime.InteropServices.Marshal.Copy(lockbits.Scan0, stream, 0, stream.Length);
            for (int y = 0; y < buffer.Height; y++)
            {
                for (int x = 0; x < buffer.Width; x++)
                {
                    int i = y * lockbits.Stride + x * 4;
                    var b = (double)stream[i + 0];
                    var g = (double)stream[i + 1];
                    var r = (double)stream[i + 2];
                    tensor[0, 0, y, x] = (float)((r - 255.0 * 0.485) / (255.0 * 0.229));
                    tensor[0, 1, y, x] = (float)((g - 255.0 * 0.456) / (255.0 * 0.224));
                    tensor[0, 2, y, x] = (float)((b - 255.0 * 0.406) / (255.0 * 0.225));
                }
            }
        }
        finally
        {
            buffer.UnlockBits(lockbits);
        }
    }
    return tensor;
}

static void CpuInference()
{
    PrintConfigure("CPU");
    var data = LoadData();
    var pred = new Dictionary<string, int>();
    using (var session = new InferenceSession(Path.Combine(ReadParameterAsStr("rootdir", "./"), "model.onnx")))
    {
        for (int i = 0; i < data.Count; i++)
        {
            var file = Path.Combine(ReadParameterAsStr("datadir", "./"), data[i].id) + ".jpg";
            var input_image = GetImageTensor(file);
            var inputTensor = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor<float>("input1", input_image),
            };
            using (var output = session.Run(inputTensor))
            {
                var result = output.ToList<DisposableNamedOnnxValue>();
                var y_pred = result[0].AsTensor<float>().ToArray();
                var y_indx = y_pred.ToList().IndexOf(y_pred.Max());
                data[i] = new(data[i].id, data[i].breed, data[i].class_id, y_indx);
            }
        }
    }
    OutputResult(data, "cpu");
}

static void GpuInference()
{
    PrintConfigure("GPU");
    var data = LoadData();
    var pred = new Dictionary<string, int>();
    using (var options = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId: 0))
    using (var session = new InferenceSession(Path.Combine(ReadParameterAsStr("rootdir", "./"), "model.onnx"), options))
    {
        for (int i = 0; i < data.Count; i++)
        {
            var file = Path.Combine(ReadParameterAsStr("datadir", "./"), data[i].id) + ".jpg";
            var input_image = GetImageTensor(file);
            var inputTensor = new List<NamedOnnxValue>()
            {
                NamedOnnxValue.CreateFromTensor<float>("input1", input_image),
            };
            using (var output = session.Run(inputTensor))
            {
                var result = output.ToList<DisposableNamedOnnxValue>();
                var y_pred = result[0].AsTensor<float>().ToArray();
                var y_indx = y_pred.ToList().IndexOf(y_pred.Max());
                data[i] = new(data[i].id, data[i].breed, data[i].class_id, y_indx);
            }
        }
    }
    OutputResult(data, "gpu");
}

static void OutputResult(List<(string id, string breed, int class_id, int predict_id)> data, string postfix)
{
    using (var writer = new StreamWriter(string.Format("result_{0}.csv", postfix), false, System.Text.Encoding.UTF8))
    {
        writer.WriteLine("id,breed,class_id,predict_id");
        foreach (var itr in data)
        {
            var (id, breed, class_id, predict_id) = itr;
            writer.WriteLine("{0},{1},{2},{3}", id, breed, class_id, predict_id);
        }
    }
}

static List<(string id, string breed, int class_id, int predict_id)> LoadData()
{
    string[] buffer;
    using (var reader = new StreamReader(Path.Combine(ReadParameterAsStr("rootdir", "./"), "sample_valid.csv")))
    {
        buffer = reader.ReadToEnd().Replace("\r\n", "\n").Split("\n");
    }
    var result = new List<(string, string, int, int)>();
    for (int i = 1; i < buffer.Length; i++)
    {
        var text = buffer[i].Trim();
        if (text.Length == 0)
        {
            continue;
        }
        var items = text.Split(",");
        result.Add(new (items[0], items[1], int.Parse(items[2]), -1));
    }
    return result;
}

if (ReadParameterAsStr("runmode", "cpu") == "gpu")
{
    // on GPU Inference
    GpuInference();
}
else
{
    // on CPU Inference
    CpuInference();
}
