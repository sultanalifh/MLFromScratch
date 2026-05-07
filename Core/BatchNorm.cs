
class BatchNorm
{
    private double _Gamma = 1;

    private double _Beta = 0;

    private double _Epsilon = 1e-5;

    public void Normalize(Vector Batch)
    {
        int Size = Batch.Length;

        double Sum = 0;


        for (int i = 0; i < Size; i++)
        {
            Sum += Batch[i];
        }

        // dU/dxi = 1 / N

        double Mean = Sum / Size;

        double Variance = 0;


        for (int i = 0; i < Size; i++)
        {
            double Diff = Batch[i] - Mean;

            Variance += Diff * Diff;
        }

        // do^2/dU = 2(xi - u) / n
        // do^2/dXi = 2(xi - u) / n

        Variance /= Size;

        // std = sqrt(o^2 + eps)

        // d std/do^2 = 1 / 2std

        // dxnorm/dstd = (dxi - u) * std^-1 = (dxi - u) * -1 * std^-2
        // = -(dxi - u) / std^2

        // dxnorm/dxi = 1 / std
        // dxnorm/du = -1 / std

        // dyi/dxi (from xNorm) = dyi / dxnorm * dxnorm/dxi
        // dyi/dxi (from variance) = dyi/dxnorm * dxnorm/dstd * dstd/do^2 * do^2/dxi
        // dyi/dxi (from var-mean) = dyi/dxnorm * dxnorm/dstd * dstd/do^2 * do^2/du * du/dxi
        // dyi/dxi (from x) = dyi/xnorm * xnorm/du * du/dxi

        // dyi/dxi (from xNorm) = dyi / dxnorm * dxnorm/dxi = Gamma / std

        // dyi/dxi (from variance) = dyi/dxnorm * dxnorm/dstd * dstd/do^2 * do^2/dxi 
        // = Gamma * -(xi - u)/std^2 * 1/2std * 2(xi - u)/n * 1/n
        // = Gamma * -(xi - u)/2std^3 * 2(xi - u)/n * 1/n
        // = Gamma * (-2(xi - u)^2)/(n^2 * 2std^3)

        // dyi/dxi (from dxi) = dyi/xnorm * xnorm/du * du/dxi
        // = Gamma * -1/std * 1/n
        // = -Gamma/nstd

        // dyi/dxnorm = gamma

        // dyi/gamma = E(dyi * dxnorm)

        // dyi/beta = E(dyi)

        for(int i = 0; i < Size; i++)
        {
            double Normalized = (Batch[i] - Mean) / Math.Sqrt(Variance + _Epsilon);

            Batch[i] = _Gamma * Normalized + _Beta;
        }
    }
}