using Gestalt.JsonCore;

enum InitType
{
    None,
    Xavier,
    He
}

class ModelBuilder
{
    public List<Layer> Layers;

    private ModelBuilder()
    {
        Layers = new List<Layer>();
    }

    public static ModelBuilder Create() => new ModelBuilder();

    public NeuralNetwork FromJson(string fileName)
    {
        Dictionary<string, object> raw_network = (Dictionary<string, object>) JsonCore.Parse(fileName)["data"];

        return FromJson(raw_network);
    }

    public NeuralNetwork FromJson(Dictionary<string, object> raw_network)
    {
        Dictionary<string, object> raw_sequential = (Dictionary<string, object>) raw_network["Sequential"];
        Dictionary<string, object> raw_optimizer = (Dictionary<string, object>) raw_network["Optimizer"];

        NeuralNetwork network = new NeuralNetwork(null);
        Sequential sequential = GetSequential(raw_sequential);
        Optimizer optimizer = GetOptimizer(sequential, raw_optimizer);

        network.Sequential = sequential;
        network.Optimizer = optimizer;
        network.Lambda = (double) raw_network.GetValueOrDefault("lambda", 0d);
        network.ModelLoss = (string) raw_network.GetValueOrDefault("LossType", "None") switch
        {
            "None" => ModelLoss.None,
            "MSE" => ModelLoss.MSE,
            "BCE" => ModelLoss.BCE,
            "CE" => ModelLoss.CE,
            _ => throw new ArgumentException($"No such loss type exists!")
        };
        
        return network;
    }

    private Optimizer GetOptimizer(Sequential sequential, Dictionary<string, object> obj)
    {
        if(sequential == null || obj == null)
        {
            return null;
        }

        string optimizerType = (string) obj["type"];
        double learningRate = (double) obj["LearningRate"];
        Optimizer optimizer;

        switch (optimizerType)
        {
            case "SGD":
                SGD sgd = new SGD(sequential, learningRate);

                optimizer = sgd;

                break;

            case "Momentum":
                List<object> raw_momentum_velocities = (List<object>) obj["Velocities"];

                List<Tensor> momentum_velocities = new List<Tensor>();

                foreach(Dictionary<string, object> raw_velocity in raw_momentum_velocities)
                {
                    momentum_velocities.Add(GetTensor(raw_velocity));
                }

                double momentum_value = (double) obj["momentum"];

                Momentum momentum = new Momentum(sequential, learningRate, momentum_value)
                {
                    Velocities = momentum_velocities
                };

                optimizer = momentum;

                break;

            case "Adam":
                List<object> raw_adam_velocities = (List<object>) obj["Velocities"];
                List<object> raw_variances = (List<object>) obj["Variances"];

                double beta1 = (double) obj["Beta1"];
                double beta2 = (double) obj["Beta2"];
                double Tbeta1 = (double) obj["TBeta1"];
                double Tbeta2 = (double) obj["TBeta2"];

                List<Tensor> adam_velocities = new List<Tensor>();
                List<Tensor> variances = new List<Tensor>();         

                foreach(Dictionary<string, object> raw_velocity in raw_adam_velocities)
                {
                    adam_velocities.Add(GetTensor(raw_velocity));
                }       

                foreach(Dictionary<string, object> raw_variance in raw_variances)
                {
                    variances.Add(GetTensor(raw_variance));
                }

                Adam adam = new Adam(sequential, learningRate, beta1, beta2)
                {
                    TBeta1 = Tbeta1,
                    TBeta2 = Tbeta2,
                    Velocities = adam_velocities,
                    Variances = variances
                };

                optimizer = adam;

                break;

            default:
                throw new ArgumentException("Invalid optimizer type for " + optimizerType);
        }

        return optimizer;
    }

    private Sequential GetSequential(Dictionary<string, object> obj)
    {
        if(obj == null)
        {
            return null;
        }

        Sequential sequential = new Sequential();

        List<Layer> layers = new List<Layer>();

        List<object> raw_layers = (List<object>) obj["Layers"];

        foreach(Dictionary<string, object> raw_layer in raw_layers)
        {
            layers.Add(GetLayer(raw_layer));
        }

        sequential.Layers = layers;

        return sequential;
    }

    private Layer GetLayer(Dictionary<string, object> obj)
    {
        Layer layer;
        
        string layerType = (string) obj["type"];
        int inputSize = Convert.ToInt32(obj["InputSize"]);
        int outputSize = Convert.ToInt32(obj["OutputSize"]);

        switch (layerType)
        {
            case "DenseLayer":
                DenseLayer dense = new DenseLayer(inputSize, outputSize);

                Dictionary<string, object> raw_weights = (Dictionary<string, object>) obj["Weights"];
                Dictionary<string, object> raw_bias = (Dictionary<string, object>) obj["Bias"];

                Parameter weight = GetParameter(raw_weights);
                Parameter bias = GetParameter(raw_bias);

                dense.Weights = weight;
                dense.Bias = bias;

                layer = dense;

                break;

            case "Conv2D":
                Dictionary<string, object> raw_filters = (Dictionary<string, object>) obj["Filters"];

                Parameter filters = GetParameter(raw_filters);

                int filterHeight = Convert.ToInt32(obj["FilterHeight"]);
                int filterWidth = Convert.ToInt32(obj["FilterWidth"]);

                int conv2d_stride = Convert.ToInt32(obj["Stride"]);
                int conv2d_padding = Convert.ToInt32(obj["Padding"]);

                Conv2D conv2D = new Conv2D(inputSize, outputSize, filterWidth, filterHeight, conv2d_stride, conv2d_padding)
                {
                    Filters = filters
                };

                layer = conv2D;

                break;

            case "MaxPool":
                int poolHeight = Convert.ToInt32(obj["PoolHeight"]);
                int poolWidth = Convert.ToInt32(obj["PoolWidth"]);

                int pool_stride = Convert.ToInt32(obj["Stride"]);
                int pool_padding = Convert.ToInt32(obj["Padding"]);

                MaxPool maxPool = new MaxPool(inputSize, outputSize, poolHeight, poolWidth, pool_stride, pool_padding);

                layer = maxPool;

                break;

            case "Flatten":

                Flatten flatten = new Flatten();

                layer = flatten;

                break;

            case "LayerNorm":
                LayerNorm layerNorm = new LayerNorm(inputSize, outputSize);

                Dictionary<string, object> raw_gamma = (Dictionary<string, object>) obj["Gamma"];
                Dictionary<string, object> raw_beta = (Dictionary<string, object>) obj["Beta"];

                Parameter gamma = GetParameter(raw_gamma);
                Parameter beta = GetParameter(raw_beta);

                layerNorm.Gamma = gamma;
                layerNorm.Beta = beta;

                layer = layerNorm;

                break;

            case "Dropout":
                double dropoutRate = (double) obj["DropoutRate"];

                Dropout dropout = new Dropout(inputSize, outputSize, dropoutRate);

                layer = dropout;

                break;

            case "ReLU":
                ReLU reLU = new ReLU();

                layer = reLU;

                break;

            case "LeakyReLU":
                LeakyReLU leakyReLU = new LeakyReLU();

                layer = leakyReLU;

                break;

            case "Sigmoid":
                Sigmoid sigmoid = new Sigmoid();

                layer = sigmoid;

                break;

            case "Tanh":
                Tanh tanh = new Tanh();

                layer = tanh;

                break;

            case "Softmax":
                Softmax softmax = new Softmax(inputSize, outputSize);

                layer = softmax;

                break;

            default:
                throw new ArgumentException("Invalid layers for: " + layerType);
        }

        return layer;
    }

    private Parameter GetParameter(Dictionary<string, object> obj)
    {
        Dictionary<string, object> raw_tensor = (Dictionary<string, object>) obj["Data"];

        string parameterName = (string) obj["Name"];

        Tensor tensor = GetTensor(raw_tensor);

        Parameter parameter = new Parameter(parameterName, tensor);

        return parameter;
    }

    private Tensor GetTensor(Dictionary<string, object> obj)
    {
        string tensorType = (string) obj["type"];

        Tensor tensor = tensorType switch
        {
            "Tensor" => GetDefaultTensor(obj),
            "Matrix" => GetMatrix(obj),
            _ => throw new ArgumentException("No such Tensor type exists!"),
        };

        return tensor;
    }

    private Tensor GetDefaultTensor(Dictionary<string, object> obj)
    {
        List<object> tensorShape = (List<object>) obj["Shape"];
        List<object> tensorData = (List<object>) obj["Data"];

        int shapeCount = tensorShape.Count;
        int dataSize = tensorData.Count;

        int[] shape = new int[shapeCount];
        double[] data = new double[dataSize];

        for(int i = 0; i < shapeCount; i++)
        {
            shape[i] = Convert.ToInt32(tensorShape[i]);
        }

        for(int i = 0; i < dataSize; i++)
        {
            data[i] = (double) tensorData[i];
        }

        Tensor tensor = new Tensor(shape)
        {
            Data = data
        };

        return tensor;
    }

    private Matrix GetMatrix(Dictionary<string, object> obj)
    {
        List<object> data = (List<object>) obj["Data"];

        int length = data.Count;

        int rows = Convert.ToInt32(obj["Rows"]);
        int cols = Convert.ToInt32(obj["Cols"]);

        if(rows * cols != length)
        {
            throw new ArgumentException("Shape mismatch!");
        }

        Matrix matrix = new Matrix(rows, cols);

        for(int row = 0; row < rows; row++)
        {
            for(int col = 0; col < cols; col++)
            {
                int idx = row * cols + col;
                
                matrix[row, col] = (double) data[idx];
            }
        }

        return matrix;
    }

    public ModelBuilder Dense(int inputSize, int outputSize, InitType init)
    {
        DenseLayer denseLayer = new DenseLayer(inputSize, outputSize);
        Parameter Weight = denseLayer.Weights;

        Initialize(init, Weight, inputSize);

        Layers.Add(denseLayer);

        return this;
    }

    public ModelBuilder Conv2D(int inputSize, int outputSize, int filterHeight, int filterWidth, int stride, int padding, InitType init)
    {
        Conv2D conv2D = new Conv2D(inputSize, outputSize, filterHeight, filterWidth, stride, padding);
        Parameter filter = conv2D.Filters;

        Initialize(init, filter, inputSize * filterHeight * filterWidth);

        Layers.Add(conv2D);

        return this;
    }

    public ModelBuilder MaxPool(int inputSize, int poolHeight, int poolWidth, int stride, int padding)
    {
        MaxPool maxPool = new MaxPool(inputSize, inputSize, poolHeight, poolWidth, stride, padding);

        Layers.Add(maxPool);

        return this;
    }

    public ModelBuilder Flatten()
    {
        Flatten flatten = new Flatten();

        Layers.Add(flatten);

        return this;
    }

    public ModelBuilder Dropout(int size, double dropoutRate)
    {
        Dropout dropout = new Dropout(size, size, dropoutRate);

        Layers.Add(dropout);

        return this;
    }

    public ModelBuilder LayerNorm(int size)
    {
        LayerNorm layerNorm = new LayerNorm(size, size);

        Layers.Add(layerNorm);

        return this;
    }

    public ModelBuilder ReLU()
    {
        ReLU reLU = new ReLU();

        Layers.Add(reLU);

        return this;
    }

    public ModelBuilder LeakyReLU()
    {
        LeakyReLU leakyReLU = new LeakyReLU();

        Layers.Add(leakyReLU);

        return this;
    }

    public ModelBuilder Tanh()
    {
        Tanh tanh = new Tanh();

        Layers.Add(tanh);

        return this;
    }

    public ModelBuilder Sigmoid()
    {
        Sigmoid sigmoid = new Sigmoid();

        Layers.Add(sigmoid);

        return this;
    }

    public ModelBuilder Softmax(int size)
    {
        Softmax softmax = new Softmax(size, size);

        Layers.Add(softmax);

        return this;
    }

    public NeuralNetwork Build()
    {
        Sequential sequential = new Sequential(Layers);

        NeuralNetwork network = new NeuralNetwork(sequential);

        Layers = new List<Layer>();

        return network;
    }

    private void Initialize(InitType init, Parameter param, int n_in)
    {
        double[] data = param.Data.Data;

        double std = init switch
        {
            InitType.Xavier => Math.Sqrt(1d / n_in),
            InitType.He => Math.Sqrt(2d / n_in),
            _ => 0.01,
        };

        for(int i = 0; i < data.Length; i++)
        {
            data[i] = Utility.RandomNormal() * std;
        }
    }
}