abstract class Layer
{
    public int InputSize;

    public int OutputSize;

    public Layer(int inputSize, int outputSize)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
    }
    public abstract Matrix Forward(Matrix x);

    public abstract Matrix Backward(Matrix x);

    public abstract void LearnGradient(double learningRate);
}