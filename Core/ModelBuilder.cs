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

                List<Matrix> momentum_velocities = new List<Matrix>();

                foreach(Dictionary<string, object> raw_velocity in raw_momentum_velocities)
                {
                    momentum_velocities.Add(GetMatrix(raw_velocity));
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

                List<Matrix> adam_velocities = new List<Matrix>();
                List<Matrix> variances = new List<Matrix>();         

                foreach(Dictionary<string, object> raw_velocity in raw_adam_velocities)
                {
                    adam_velocities.Add(GetMatrix(raw_velocity));
                }       

                foreach(Dictionary<string, object> raw_variance in raw_variances)
                {
                    variances.Add(GetMatrix(raw_variance));
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
                ReLU reLU = new ReLU(inputSize, outputSize);

                layer = reLU;

                break;

            case "LeakyReLU":
                LeakyReLU leakyReLU = new LeakyReLU(inputSize, outputSize);

                layer = leakyReLU;

                break;

            case "Sigmoid":
                Sigmoid sigmoid = new Sigmoid(inputSize, outputSize);

                layer = sigmoid;

                break;

            case "Tanh":
                Tanh tanh = new Tanh(inputSize, outputSize);

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
        Dictionary<string, object> raw_matrix = (Dictionary<string, object>) obj["Data"];

        string parameterName = (string) obj["Name"];

        Matrix matrix = GetMatrix(raw_matrix);

        Parameter parameter = new Parameter(matrix, parameterName);

        return parameter;
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

        Initialize(init, Weight);

        Layers.Add(denseLayer);

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

    public ModelBuilder ReLU(int size)
    {
        ReLU reLU = new ReLU(size, size);

        Layers.Add(reLU);

        return this;
    }

    public ModelBuilder LeakyReLU(int size)
    {
        LeakyReLU leakyReLU = new LeakyReLU(size, size);

        Layers.Add(leakyReLU);

        return this;
    }

    public ModelBuilder Tanh(int size)
    {
        Tanh tanh = new Tanh(size, size);

        Layers.Add(tanh);

        return this;
    }

    public ModelBuilder Sigmoid(int size)
    {
        Sigmoid sigmoid = new Sigmoid(size, size);

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

    private void Initialize(InitType init, Parameter param)
    {
        int inputSize = param.Data.Cols;
        int outputSize = param.Data.Rows;

        double std = init switch
        {
            InitType.Xavier => Math.Sqrt(1d / inputSize),
            InitType.He => Math.Sqrt(2d / inputSize),
            _ => 0.01,
        };

        for(int i = 0; i < outputSize; i++)
        {
            for(int j = 0; j < inputSize; j++)
            {
                param.Data[i,j] = Utility.RandomNormal() * std;
            }
        }
    }
}