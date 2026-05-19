enum IDXObject
{
    uByte = 0,
    sByte,
    Short,
    Int,
    Float,
    Double
}

static class IDXParser
{
    public static object Parse(string fileName, List<int> limiter)
    {
        using Stream stream = File.OpenRead(fileName);

        using BinaryReader binaryReader = new BinaryReader(stream);

        byte[] magic = binaryReader.ReadBytes(4);

        byte objectByte = magic[2];
        byte numberOfDimension = magic[3];

        IDXObject idxObject = GetObject(objectByte);

        List<int> dimensions = new List<int>();

        if(limiter == null)
        {
            limiter = new List<int>();
        }

        for(int i = 0; i < numberOfDimension; i++)
        {
            byte[] bytes = binaryReader.ReadBytes(4);

            Array.Reverse(bytes);

            int dimension = BitConverter.ToInt32(bytes);

            dimensions.Add(dimension);

            if(limiter.Count == i)
            {
                limiter.Add(dimension);
            }
        }
        
        object result = RecurseDimension(binaryReader, idxObject, dimensions, limiter);

        return result;
    }

    private static object RecurseDimension(BinaryReader reader, IDXObject idxObject, List<int> dimensions, List<int> limiter, int idx = 0)
    {
        int numberOfDimension = dimensions.Count;

        if(idx >= numberOfDimension)
        {
            byte[] data = ReadByte(reader, idxObject);

            return ParseByte(data, idxObject);
        }

        int dimensionSize = Math.Min(limiter[idx], dimensions[idx]);

        List<object> result = new List<object>();

        for(int i = 0; i < dimensionSize; i++)
        {
            object child = RecurseDimension(reader, idxObject, dimensions, limiter, idx + 1);
            
            result.Add(child);
        }

        return result;
    }

    public static IDXObject GetObject(byte Byte) => Byte switch
    {
        0x08 => IDXObject.uByte,
        0x09 => IDXObject.sByte,
        0x0B => IDXObject.Short,
        0x0C => IDXObject.Int,
        0x0D => IDXObject.Float,
        0x0E => IDXObject.Double,
        _ => throw new ArgumentException("Invalid object")
    };

    public static byte[] ReadByte(BinaryReader reader, IDXObject idxObject) => idxObject switch
    {
        IDXObject.uByte => reader.ReadBytes(1),
        IDXObject.sByte => reader.ReadBytes(1),
        IDXObject.Short => reader.ReadBytes(2),
        IDXObject.Int => reader.ReadBytes(4),
        IDXObject.Float => reader.ReadBytes(4),
        IDXObject.Double => reader.ReadBytes(8),
        _ => throw new ArgumentException("No object specified!")
    };

    public static object ParseByte(byte[] bytes, IDXObject idxObject) 
    {
        if(BitConverter.IsLittleEndian && bytes.Length > 1)
        {
            Array.Reverse(bytes);
        }

        return idxObject switch
        {  
            IDXObject.uByte     => (object) bytes[0],
            IDXObject.sByte     => (object) (sbyte) bytes[0],
            IDXObject.Short     => (object) BitConverter.ToInt16(bytes),
            IDXObject.Int       => (object) BitConverter.ToInt32(bytes),
            IDXObject.Float     => (object) BitConverter.ToSingle(bytes),
            IDXObject.Double    => (object) BitConverter.ToDouble(bytes),
            _ => throw new ArgumentException("No object specified!")
        };
    }
}