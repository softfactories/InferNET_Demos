using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace sf.infernet.demos
{
    class Program
    {
        static void Main(string[] args)
        {
            Experiment_1();
        }

        private static void Experiment_1()
        {
            // PM erstellen
            Variable<bool> ersteMünzeKopf = Variable.Bernoulli(0.5);
            Variable<bool> zweiteMünzeKopf = Variable.Bernoulli(0.5);
            Variable<bool> beideMünzenKopf = ersteMünzeKopf & zweiteMünzeKopf;
            
            // Inferenz-Engine (IE) erstellen
            InferenceEngine engine = new InferenceEngine();
  
            // Inferenz ausführen
            var ergebnis = engine.Infer<Bernoulli>(beideMünzenKopf);
            double beideMünzenZeigenKöpfe = ergebnis.GetProbTrue();


            Console.WriteLine("Die Wahrscheinlichkeit - beide Münzen " +
                "zeigen Köpfe: {0}", beideMünzenZeigenKöpfe);
        }
    }
}
