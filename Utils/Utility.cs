public static class Utility
{
    public static Random Random = new Random();

    public static double e5Eps = 1e-5;

    public static double e12Eps = 1e-12;

    public static double RandomNormal()
    {
        double u1 = Random.NextDouble();
        double u2 = Random.NextDouble();
        double z = Math.Sqrt(-2*Math.Log(u1)) * Math.Cos(2*Math.PI*u2);
        return z;
    }

    public static int GCD(int a, int b) => b > a ? GCD(b,a) : a % b == 0 ? b : GCD(b, a%b);
}