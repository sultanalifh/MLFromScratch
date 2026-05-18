using System.Text.Json.Serialization;
using Gestalt.JsonCore;

class ModelCheckpoint
{
    [JsonInclude]
    public NeuralNetwork NeuralNetwork;
    [JsonInclude]
    public string ModelName;
    [JsonInclude]
    public string FileName;
    [JsonInclude]
    public double BestValidationLoss;
    [JsonInclude]
    public int CurrentEpoch;
    [JsonInclude]
    public int BestEpoch;

    public void PrintStats()
    {
        Console.WriteLine("--- Model Stats ---");
        Console.WriteLine($"Model name: {ModelName}");
        Console.WriteLine($"Current epoch: " + CurrentEpoch);
        Console.WriteLine($"Best Epoch: {BestEpoch}");
        Console.WriteLine($"Best validation loss: {BestValidationLoss}");
        Console.WriteLine();
    }
    
    public void Track(double validationLoss)
    {
        if(validationLoss < BestValidationLoss)
        {
            BestValidationLoss = validationLoss;
            BestEpoch = CurrentEpoch;

            Console.WriteLine("New best model found!");
            Console.WriteLine("Validation loss: " + BestValidationLoss);
            Console.WriteLine("Saving checkpoint...");
            Save();
            Console.WriteLine("Checkpoint saved!");
            Console.WriteLine();
        }
    }

    public void Track() => ++CurrentEpoch;

    public void Save() => JsonCore.Serialize(this, FileName);

    public static ModelCheckpoint Load(string fileName)
    {
        NeuralNetwork network;
        double bestValidationLoss = int.MaxValue;
        int currentEpoch = 0;
        int bestEpoch = 0;
        string modelName = "";

        if (!File.Exists(fileName))
        {
            Console.WriteLine("Model doesn't Exists!");
            Console.WriteLine("Creating new one...");
            Console.WriteLine("What name do you call this model: ");
            modelName = Console.ReadLine();
            network = Create();

            Console.WriteLine("Model created!");
            Console.WriteLine();
        }
        else
        {
            Console.WriteLine("Model found!");
            Console.WriteLine();

            Dictionary<string, object> raw_checkpoint = (Dictionary<string, object>) JsonCore.Parse(fileName)["data"];
            Dictionary<string, object> raw_network = (Dictionary<string, object>) raw_checkpoint["NeuralNetwork"];

            network = ModelBuilder.Create().FromJson(raw_network);
            modelName = (string) raw_checkpoint["ModelName"];
            bestValidationLoss = (double) raw_checkpoint["BestValidationLoss"];
            currentEpoch = Convert.ToInt32(raw_checkpoint["CurrentEpoch"]);
            bestEpoch = Convert.ToInt32(raw_checkpoint["BestEpoch"]);
        }

        return new ModelCheckpoint()
        {
            NeuralNetwork = network,
            ModelName = modelName,
            FileName = fileName,
            BestValidationLoss = bestValidationLoss,
            CurrentEpoch = currentEpoch,
            BestEpoch = bestEpoch
        };
    }

    public static NeuralNetwork Create()
    {
        NeuralNetwork baseNetwork = ModelBuilder.Create()
            .Dense(2, 64, InitType.He)
            .LayerNorm(64)
            .LeakyReLU(64)
            .Dropout(64, 0.05)

            .Dense(64, 64, InitType.He)
            .LayerNorm(64)
            .LeakyReLU(64)
            .Dropout(64, 0.05)

            .Dense(64, 64, InitType.He)
            .LayerNorm(64)
            .LeakyReLU(64)

            .Dense(64, 3, InitType.Xavier)
            .Softmax(3)
            .Build();


        Optimizer optimizer = baseNetwork.AdamOptimizer(0.003);
        Sequential sequential = baseNetwork.Sequential;
        baseNetwork.ModelLoss = ModelLoss.CE;
        baseNetwork.Lambda = 0.0005;

        return baseNetwork;
    }
}