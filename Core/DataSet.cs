
class DataSet
{
    private Vector _Data;

    private double _YTrue;

    public Vector Data
    {
        get
        {
            return _Data;
        }
    }

    public double Ytrue
    {
        get
        {
            return _YTrue;
        }
    }

    public DataSet(Vector Data, double Ytrue)
    {
        _Data = Data;
        _YTrue = Ytrue;
    }
}