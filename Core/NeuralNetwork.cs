using System.Text.Json.Serialization;

class NeuralNetwork
{
    [JsonInclude]
    public Sequential Sequential;
    [JsonInclude]
    public Optimizer Optimizer;
    public ModelLoss ModelLoss = ModelLoss.None;

    public string LossType
    {
        get
        {
            return Enum.GetName(typeof(ModelLoss), ModelLoss);
        }
    }
    private double _Lambda = 0;
    public double Lambda
    {
        get
        {
            return _Lambda;
        }
        set
        {
            foreach(Layer layer in Sequential.Layers)
            {
                if(layer is DenseLayer)
                {
                    DenseLayer denseLayer = (DenseLayer) layer;
                    denseLayer.Lambda = value;
                }
            }

            _Lambda = value;
        }
    }


    public NeuralNetwork(Sequential sequential)
    {
        Sequential = sequential;
    }

    public NeuralNetwork(Sequential sequential, Optimizer optimizer)
    {
        Sequential = sequential;
        Optimizer = optimizer;
    }

    public Tensor Predict(Tensor x) => Sequential.Forward(x);

    public void Fit(Tensor x, Tensor yTrue)
    {
        Optimizer.ZeroGrad();

        foreach(Layer layer in Sequential.Layers)
        {
            if(layer is Dropout)
            {
                Dropout dropout = (Dropout) layer;
                dropout.IsTraining = true;
            }
        }

        Tensor yPred = Sequential.Forward(x);

        Tensor grad = Loss.LossGrad(yPred, yTrue, ModelLoss);

        Sequential.Backward(grad);

        foreach(Layer layer in Sequential.Layers)
        {
            if(layer is Dropout)
            {
                Dropout dropout = (Dropout) layer;
                dropout.IsTraining = false;
            }
        }

        Optimizer.Step();
    }

    public Optimizer SGDOptimizer(double learningRate) => Optimizer = new SGD(Sequential, learningRate);

    public Optimizer MomentumOptimizer(double learningRate, double momentum = 0.9) => Optimizer = new Momentum(Sequential, learningRate, momentum);

    public Optimizer AdamOptimizer(double learningRate, double beta1 = 0.9, double beta2 = 0.999) => Optimizer = new Adam(Sequential, learningRate, beta1, beta2);
}