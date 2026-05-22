using System.Text.Json.Serialization;

class Tensor
{
    [JsonInclude]
    public int[] Shape;
    
    [JsonInclude]
    public double[] Data;

    public double this[params int[] indexer]
    {
        get
        {
            int flattenIndex = GetFlatIndex(indexer);

            return Data[flattenIndex];
        }
        set
        {
            int flattenIndex = GetFlatIndex(indexer);

            Data[flattenIndex] = value;
        }
    }

    public Tensor(params int[] shape)
    {
        Shape = shape;

        int flatSize = GetFlatSize(shape);

        Data = new double[flatSize];
    }

    public int GetFlatIndex(int[] indexer)
    {
        int shapeDimensionCount = Shape.Length;
        int dimensionCount = indexer.Length;

        if(shapeDimensionCount != dimensionCount)
        {
            throw new ArgumentException("Number of dimension was not the same!");
        }

        int flattenIndex = 0;

        int stackedIndex = 1;

        for(int i = shapeDimensionCount - 1; i >= 0; i--)
        {
            if(indexer[i] < 0 || indexer[i] >= Shape[i])
            {
                throw new ArgumentOutOfRangeException("Index was out of bounds!");
            }

            flattenIndex += indexer[i] * stackedIndex;

            stackedIndex *= Shape[i];
            
        }

        return flattenIndex;
    }

    public int[] GetIndices(int flatIndex)
    {
        int shapeLength = Shape.Length;
        int[] indices = new int[shapeLength];

        for(int i = shapeLength - 1; i >= 0; i--)
        {
            indices[i] = flatIndex % Shape[i];
            
            flatIndex /= Shape[i];
        }

        return indices;
    }
    public Tensor Reshape(params int[] newShape)
    {
        int flattenSize = GetFlatSize(Shape);
        int flattenNewSize = GetFlatSize(newShape);

        if(flattenNewSize != flattenSize)
        {
            throw new ArgumentException("Could'nt reshape into smaller form!");
        }

        Tensor tensor = new Tensor(newShape);

        for(int i = 0; i < flattenSize; i++)
        {
            tensor.Data[i] = Data[i];
        }

        return tensor;
    }

    public Tensor Dot(Tensor other)
    {
        if(other.Shape.Length != 2 && this.Shape.Length != 2)
        {
            throw new ArgumentException("Dot was not possible between non-matrices tensor!");
        }

        int otherRows = other.Shape[0];
        int otherCols = other.Shape[1];

        if(Shape[1] != otherRows)
        {
            throw new ArgumentException("Dot was not possible!");
        }

        Tensor result = new Tensor(Shape[0], otherCols);

        for(int i = 0; i < Shape[0]; i++)
        {
            for(int j = 0; j < otherCols; j++)
            {
                for(int k = 0; k < Shape[1]; k++)
                {
                    result[i,j] += this[i,k] * other[k,j];
                }
            }
        }

        return result;
    }

    public Tensor Transpose()
    {
        if(Shape.Length != 2)
        {
            throw new ArgumentException("Cannot transpose non-matrices");
        }

        int rows = Shape[0];
        int cols = Shape[1];

        Tensor result = new Tensor(cols, rows);

        for(int i = 0; i < rows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                result[j,i] = this[i,j];
            }
        }

        return result;
    }

    public virtual Tensor Clone()
    {
        int dimensionCount = Shape.Length;
        int flattenSize = Data.Length;

        int[] clonedShape = new int[dimensionCount];

        Array.Copy(Shape, clonedShape, dimensionCount);

        Tensor tensor = new Tensor(clonedShape);

        for(int i = 0; i < flattenSize; i++)
        {
            tensor.Data[i] = Data[i];
        }

        return tensor;
    }

    public Tensor Apply(Func<double, double> func)
    {
        int flattenSize = Data.Length;

        Tensor tensor = Clone();

        for(int i = 0; i < flattenSize; i++)
        {
            tensor.Data[i] = func.Invoke(tensor.Data[i]);
        }

        return tensor;
    }

    public Tensor Pad(int padding)
    {
        if(padding == 0)
        {
            return this;
        }

        int shapeLength = Shape.Length;

        if(shapeLength < 2)
        {
            throw new ArgumentException("Could'nt pad Tensor with lesser dimension than 2");
        }

        int[] newShape = new int[shapeLength];

        Array.Copy(Shape, newShape, shapeLength);

        newShape[shapeLength - 1] += padding * 2;
        newShape[shapeLength - 2] += padding * 2;

        Tensor tensor = new Tensor(newShape);
        
        int dataSize = tensor.Data.Length;

        for(int i = 0; i < dataSize; i++)
        {
            int[] indices = tensor.GetIndices(i);

            int dx = indices[shapeLength - 1] - padding;
            int dy = indices[shapeLength - 2] - padding;

            if(dx >= 0 && dx < Shape[shapeLength - 1] && dy >= 0 && dy < Shape[shapeLength - 2])
            {
                indices[shapeLength - 1] = dx;
                indices[shapeLength - 2] = dy;

                tensor.Data[i] = this[indices];
            }
        }

        return tensor;
    }

    public bool DimensionMatch(Tensor other)
    {
        int[] otherShape = other.Shape;

        int thisDim = Shape.Length;
        int otherDim = otherShape.Length;

        return thisDim == otherDim;
    }

    public bool ShapeMatch(Tensor other)
    {
        if (!DimensionMatch(other))
        {
            throw new ArgumentException("Dimension was not match!");
        }

        int[] otherShape = other.Shape;
        int otherDim = otherShape.Length;

        for(int i = 0; i < otherDim; i++)
        {
            if(Shape[i] != otherShape[i])
            {
                return false;
            }
        }

        return true;
    }

    public static int GetFlatSize(int[] shape)
    {
        int flattenSize = 1;

        foreach(int dimensionSize in shape)
        {
            flattenSize *= dimensionSize;
        }

        return flattenSize;
    }
}