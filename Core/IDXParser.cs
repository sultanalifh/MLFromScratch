enum IDXObject
{
    uByte,
    sByte,
    Short,
    Int,
    Float,
    Double
}

static class IDXParser
{
    public static object Parse(string fileName)
    {
        using Stream stream = File.OpenRead(fileName);

        using BinaryReader binaryReader = new BinaryReader(stream);

        byte[] magic = binaryReader.ReadBytes(4);

        byte objectByte = magic[2];
        byte numberOfDimension = magic[3];

        IDXObject Object = GetObject(objectByte);

        List<int> dimensions = new List<int>();

        for(int i = 0; i < numberOfDimension; i++)
        {
            byte[] bytes = binaryReader.ReadBytes(4);

            Array.Reverse(bytes);

            int dimension = BitConverter.ToInt32(bytes);

            dimensions.Add(dimension);
        }
        
        object result = RecurseDimension(binaryReader, Object, dimensions);

        return result;
    }

    private static object RecurseDimension(BinaryReader reader, IDXObject Object, List<int> dimensions, int idx = 0)
    {
        int numberOfDimension = dimensions.Count;

        if(idx >= numberOfDimension)
        {
            byte[] data = ReadByte(reader, Object);

            return ParseByte(data, Object);
        }

        int dimensionSize = dimensions[idx];

        List<object> result = new List<object>();

        for(int i = 0; i < dimensionSize; i++)
        {
            object child = RecurseDimension(reader, Object, dimensions, idx + 1);
            
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

    public static byte[] ReadByte(BinaryReader reader, IDXObject Object) => Object switch
    {
        IDXObject.uByte => reader.ReadBytes(1),
        IDXObject.sByte => reader.ReadBytes(1),
        IDXObject.Short => reader.ReadBytes(2),
        IDXObject.Int => reader.ReadBytes(4),
        IDXObject.Float => reader.ReadBytes(4),
        IDXObject.Double => reader.ReadBytes(8),
        _ => throw new ArgumentException("No object specified!")
    };

    public static dynamic ParseByte(byte[] bytes, IDXObject Object) 
    {
        if(BitConverter.IsLittleEndian && bytes.Length > 1)
        {
            Array.Reverse(bytes);
        }

        return Object switch
        {  
            IDXObject.uByte => bytes[0],
            IDXObject.sByte => (sbyte) bytes[0],
            IDXObject.Short => BitConverter.ToInt16(bytes),
            IDXObject.Int => BitConverter.ToInt32(bytes),
            IDXObject.Float => BitConverter.ToSingle(bytes),
            IDXObject.Double => BitConverter.ToDouble(bytes),
            _ => throw new ArgumentException("No object specified!")
        };
    }
}