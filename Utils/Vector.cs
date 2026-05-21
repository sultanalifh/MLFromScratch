class Vector : Tensor
{
    public int Length => Shape[0];

    public Vector(int length) : base(length)
    {
        
    }
}