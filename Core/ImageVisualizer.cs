static class ImageVisualizer
{
    public static void Print(double[] pixels, int width, bool percentage = false)
    {
        int n = pixels.Length;

        for(int i = 0; i < n; i++)
        {
            if(i % width == 0) Console.WriteLine();

            double pixel = percentage ? pixels[i] * 255 : pixels[i];

            char stroke = pixel switch
            {
                < 50 => '.',
                < 100 => '-',
                < 150 => '*',
                < 200 => 'O',
                _ => '#'
            };

            Console.Write(stroke);
        }
        
        Console.WriteLine();
    }

    public static Tensor normalize(Tensor image)
    {
        int n = image.Data.Length;

        Tensor result = new Tensor(n);

        for(int i = 0; i < n; i++)
        {
            result[i] = image.Data[i] / 255.0;
        }

        return result;
    }
    public static double[] Translate(double[] image, int width, int dx, int dy)
    {
        dy *= -1;

        int n = image.Length;

        int height = n / width;

        double[] result = new double[n];

        for(int i = 0; i < height; i++)
        {
            for(int j = 0; j < width; j++)
            {
                int x = j - dx;
                int y = i - dy;

                if(y < 0 || y >= height || x < 0 || x >= width)
                {
                    result[i * width  + j] = 0;
                }
                else result[i * width + j] = image[y * width + x];
            }
        }

        return result;
    }
    public static double[] Scale(double[] image, int width, int scale)
    {
        int n = image.Length;
        
        int cx = width / 2;
        int cy = width / 2;

        double[] result = new double[n];

        for(int y = 0; y < width; y++)
        {
            for(int x = 0; x < width; x++)
            {
                double srcX = (x - cx) / scale + cx;
                double srcY = (y - cy) / scale + cy;

                result[y * width + x] = BiliniarInterpolation(image, width, srcX, srcY);
            }
        }

        return result;
    }

    public static double[] Rotate(double[] image, int width, double theta)
    {
        int n = image.Length;

        int cx = width / 2;
        int cy = width / 2;

        double[] result = new double[n];

        for(int y = 0; y < width; y++)
        {
            for(int x = 0; x < width; x++)
            {
                int dx = x - cx;
                int dy = y - cy;

                double srcX = Math.Cos(theta) * dx + Math.Sin(theta) * dy + cx;
                double srcY = -Math.Sin(theta) * dx + Math.Cos(theta) * dy + cy;

                result[y * width + x] = BiliniarInterpolation(image, width, srcX, srcY);
            }
        }

        return result;
    }

    public static double BiliniarInterpolation(double[] image, int width, double x, double y)
    {
        int x0 = (int) x;
        int y0 = (int) y;

        int x1 = x0 + 1;
        int y1 = y0 + 1;

        double tx = x - x0;
        double ty = y - y0;

        double ldy0, ldy1, rdy0, rdy1;

        ldy0 = ldy1 = rdy0 = rdy1 = 0;

        if(y0 >= 0 && y0 < width)
        {
            ldy0 = x0 >= 0 && x0 < width ? image[y0 * width + x0] : 0;
            ldy1 = x1 >= 0 && x1 < width ? image[y0 * width + x1] : 0;
        }

        if(y1 >= 0 && y1 < width)
        {
            rdy0 = x0 >= 0 && x0 < width ? image[y1 * width + x0] : 0;
            rdy1 = x1 >= 0 && x1 < width ? image[y1 * width + x1] : 0;
        }

        double ldy = ldy0 * (1 - tx) + ldy1 * tx;
        double rdy = rdy0 * (1 - tx) + rdy1 * tx;

        double result = ldy * (1 - ty) * rdy * ty;

        return result;
    }
}