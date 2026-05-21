using System.Drawing;
using System.Drawing.Imaging;

static class DecisionBoundary
{
    public static void SaveSpiralDatasetBoundaryImage(NeuralNetwork model, string filename)
    {
        int Width = 300;
        int Height = 300;
        int total = Width * Height;

        double min = -2;
        double max = 2;
        double diff = max - min;

        Color[] color = [Color.Red, Color.Green, Color.Blue];

        Matrix testcase = new Matrix(total, 2);

        Bitmap bitmap = new Bitmap(Width, Height, PixelFormat.Format32bppArgb);

        for(int i = 0; i < total; i++)
        {
            int x = i % Width;
            int y = i / Width;

            double px = x * diff / Width + min;
            double py = y * diff / Height + min;

            testcase[i,0] = px;
            testcase[i,1] = py;
        }

        int batchSize = testcase.Rows;
        int numClass = 3;

        Tensor yPred = model.Predict(testcase);
        
        for(int i = 0; i < total; i++)
        {
            int x = i % Width;
            int y = i / Width;

            int maxProb = 0;

            for(int j = 0; j < numClass; j++)
            {
                if(yPred[i,j] > yPred[i, maxProb])
                {
                    maxProb = j;
                }
            }

            bitmap.SetPixel(x, y, color[maxProb]);
        }

        bitmap.Save(filename, ImageFormat.Jpeg);

        bitmap.Dispose();
    }
}