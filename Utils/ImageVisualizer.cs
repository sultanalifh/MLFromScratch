static class ImageVisualizer
{
    public static void Print(double[] pixels, int width, bool percentage = false)
    {
        int n = pixels.Length;

        for(int i = 0; i < n; i++)
        {
            if(i % width == 0) Console.WriteLine();

            char stroke;
            
            if((percentage && pixels[i] * 255 < 50) || pixels[i] < 50)
                stroke = '.';
            else if((percentage && pixels[i] * 255 < 100) || pixels[i] < 100)
                stroke = '-';
            else if((percentage && pixels[i] * 255 < 150) || pixels[i] < 150)
                stroke = '*';
            else if((percentage && pixels[i] * 255 < 200) || pixels[i] < 200)
                stroke = 'O';
            else stroke = '#';

            Console.Write(stroke);
        }
    }

    public static double[] normalize(double[] image)
    {
        int n = image.Length;

        double[] result = new double[n];

        for(int i = 0; i < n; i++)
        {
            result[i] = result[i] / 255.0;
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
}