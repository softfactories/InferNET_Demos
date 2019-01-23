using System;
using Microsoft.ML.Probabilistic.Models;
using Microsoft.ML.Probabilistic.Distributions;

namespace sf.infernet.demos
{
    class Program
    {
        static void Main(string[] args)
        {
            Experiment_2();
        }

        private static void Experiment_2()
        {
            // Wahrscheinlichkeit von 6 Würfelaugen ist 1 zu 6 = 1/6 ca 0,17
            double erfolgWurfel = 0.17;
            // PM erstellen
            Variable<bool> ersterWürfelWurf = Variable.Bernoulli(erfolgWurfel);
            Variable<bool> zweiterWürfelWurf = Variable.Bernoulli(erfolgWurfel);
            Variable<bool> beideWürfelWurf = ersterWürfelWurf & zweiterWürfelWurf;

            // Inferenz-Engine (IE) erstellen
            InferenceEngine engine = new InferenceEngine();
#if SHOW_MODEL
            engine.ShowFactorGraph = true; // PM visualisieren
#endif

            // 1. Inferenz ausführen - beide Münzen zeigen Köpfe
            var ergebnis1 = engine.Infer<Bernoulli>(beideWürfelWurf);
            double beideWürfelZeigenSechs = ergebnis1.GetProbTrue();
            var ergebnis2 = engine.Infer<Bernoulli>(ersterWürfelWurf);
            double ersterWürfelZeigtSechs = ergebnis2.GetProbTrue();

            showResult("Prior", beideWürfelZeigenSechs, ersterWürfelZeigtSechs);

            // Beobachtung - beide Würfel zeigen keine "6" gleichzeitig
            beideWürfelWurf.ObservedValue = false;
            Console.WriteLine("\nBeobachtung: beide Würfel zeigen \"6\"={0}\n"
                , beideWürfelWurf.ObservedValue);

            // 2. Inferenz ausführen - beide Münzen zeigen Köpfe
            var ergebnis3 = engine.Infer<Bernoulli>(beideWürfelWurf);
            beideWürfelZeigenSechs = ergebnis3.GetProbTrue();
            var ergebnis4 = engine.Infer<Bernoulli>(ersterWürfelWurf);
            ersterWürfelZeigtSechs = ergebnis4.GetProbTrue();

            showResult("Posterior", beideWürfelZeigenSechs, ersterWürfelZeigtSechs);
        }

        private static void showResult(string prefix,
            double beideWürfelZeigenSechs, double ersterWürfelZeigtSechs)
        {
            Console.WriteLine("{1}: Beide Würfel zeigen \"6\": {0}" , beideWürfelZeigenSechs, prefix);
            Console.WriteLine("{1}: 1. Würfel zeigt \"6\": {0}", ersterWürfelZeigtSechs, prefix);
        }
    }
}
