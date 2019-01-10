using System;
using Microsoft.ML.Probabilistic.Models;

namespace sf.infernet.demos
{
    class Program
    {
        static void Main(string[] args)
        {
            Variable<bool> ersteMünze = Variable.Bernoulli(0.5);
            Variable<bool> zweiteMünze = Variable.Bernoulli(0.5);
            Variable<bool> beideMünzen = ersteMünze & zweiteMünze;

            InferenceEngine engine = new InferenceEngine();

#if SHOW_MODEL
            engine.ShowFactorGraph = true;
#endif

            Console.WriteLine("Die Wahrscheinlichkeit - beide Münzen zeigen Köpfe: "
                + engine.Infer(beideMünzen));
        }
    }
}
