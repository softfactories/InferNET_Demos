﻿using System;
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
            // PM erstellen
            Variable<bool> ersteMünzeWurf = Variable.Bernoulli(0.5);
            Variable<bool> zweiteMünzeWurf = Variable.Bernoulli(0.5);
            Variable<bool> beideMünzenWurf = ersteMünzeWurf & zweiteMünzeWurf;

            // Inferenz-Engine (IE) erstellen
            InferenceEngine engine = new InferenceEngine();
#if SHOW_MODEL
            engine.ShowFactorGraph = true; // PM visualisieren
#endif

            // 1. Inferenz ausführen - beide Münzen zeigen Köpfe
            var ergebnis1 = engine.Infer<Bernoulli>(beideMünzenWurf);
            double beideMünzenZeigenKöpfe = ergebnis1.GetProbTrue();
            var ergebnis2 = engine.Infer<Bernoulli>(ersteMünzeWurf);
            double ersteMünzeZeigtKopf = ergebnis2.GetProbTrue();

            Console.WriteLine("Prior: Beide Münzen zeigen Köpfe: {0}"
                , beideMünzenZeigenKöpfe);
            Console.WriteLine("Prior: 1. Münze zeigt Kopf: {0}"
                , ersteMünzeZeigtKopf);

            // Beobachtung - beide Münzen zeigen Köpfe nicht
            beideMünzenWurf.ObservedValue = false;
            Console.WriteLine("\nBeobachtung: beide Münzen zeigen Köpfe = {0}\n"
                , beideMünzenWurf.ObservedValue);

            // 2. Inferenz ausführen - beide Münzen zeigen Köpfe
            var ergebnis3 = engine.Infer<Bernoulli>(beideMünzenWurf);
            beideMünzenZeigenKöpfe = ergebnis3.GetProbTrue();
            var ergebnis4 = engine.Infer<Bernoulli>(ersteMünzeWurf);
            ersteMünzeZeigtKopf = ergebnis4.GetProbTrue();

            Console.WriteLine("Posterior: P(beide Münzen zeigen Köpfe)={0}", 
                beideMünzenZeigenKöpfe);
            Console.WriteLine("Posterior: P(1. Münze zeigt Kopf)={0}", 
                ersteMünzeZeigtKopf);
        }
    }
}
