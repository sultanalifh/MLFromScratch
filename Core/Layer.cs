abstract class Layer
{
    protected abstract Matrix Forward(Matrix x);

    protected abstract Matrix Backward(Matrix x);
}