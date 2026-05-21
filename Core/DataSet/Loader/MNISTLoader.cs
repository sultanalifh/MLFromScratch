static class MNISTLoader
{
    public static MNISTDataset Load(string fileName, int numSamples, int width = 28)
    {
        int imageSize = width * width;

        MNISTDataset dataset = new MNISTDataset();

        List<int> limiter = new List<int>() {numSamples};

        List<object> raw_images = (List<object>) IDXParser.Parse($"{fileName}-images.idx3-ubyte", limiter);
        List<object> raw_labels = (List<object>) IDXParser.Parse($"{fileName}-labels.idx1-ubyte", limiter);

        for(int i = 0; i < numSamples; i++)
        {
            List<object> raw_image = (List<object>) raw_images[i];

            Tensor image = new Tensor(imageSize);
            int label = (byte) raw_labels[i];

            for(int j = 0; j < width; j++)
            {
                List<object> pixels = (List<object>) raw_image[j];

                for(int k = 0; k < width; k++)
                {
                    int idx = j * width + k;

                    image[idx] = (byte) pixels[k];
                }
            }

            image = ImageVisualizer.normalize(image);

            MNISTSample sample = new MNISTSample(image, label, width);

            dataset.Samples.Add(sample);
        }

        return dataset;
    }

}