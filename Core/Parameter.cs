class Parameter
{
    public string Name;
    public Matrix Data;
    public Matrix Grad;

    public Parameter(Matrix data, string name)
    {
        Name = name;
        Data = data;
        Grad = new Matrix(Data.Rows, Data.Cols);
    }

    public Parameter(int rows, int cols, string name)
    {
        Name = name;
        Data = new Matrix(rows, cols);
        Grad = new Matrix(rows, cols);
    }
}