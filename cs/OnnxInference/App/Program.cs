using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

const int ImageSize = 224;
const string RootDir = "../../../../../../../python/";
const string DataDir = "../../../../../../../python/data/dog-breed-identification/train/";

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
    var tensor = new DenseTensor<float>(new[] { 1, 3, ImageSize, ImageSize });
    using (var bitmap = new Bitmap(filepath))
    using (var buffer = ImageResize(bitmap, ImageSize, ImageSize))
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
                    var b = (float)stream[i + 0];
                    var g = (float)stream[i + 1];
                    var r = (float)stream[i + 2];
                    tensor[0, 0, y, x] = (r - 255.0f * 0.485f) / (255.0f * 0.229f);
                    tensor[0, 1, y, x] = (g - 255.0f * 0.456f) / (255.0f * 0.224f);
                    tensor[0, 2, y, x] = (b - 255.0f * 0.406f) / (255.0f * 0.225f);
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
    var data = LoadData();
    var pred = new Dictionary<string, int>();
    using (var session = new InferenceSession(Path.Combine(RootDir, "model.onnx")))
    {
        for (int i = 0; i < data.Count; i++)
        {
            var file = Path.Combine(DataDir, data[i].id) + ".jpg";
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
    var data = LoadData();
    var pred = new Dictionary<string, int>();
    using (var options = SessionOptions.MakeSessionOptionWithCudaProvider(deviceId: 0))
    using (var session = new InferenceSession(Path.Combine(RootDir, "model.onnx"), options))
    {
        for (int i = 0; i < data.Count; i++)
        {
            var file = Path.Combine(DataDir, data[i].id) + ".jpg";
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
    using (var reader = new StreamReader(Path.Combine(RootDir, "sample_valid.csv")))
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

// on CPU Inference
//CpuInference();

// on GPU Inference
GpuInference();
