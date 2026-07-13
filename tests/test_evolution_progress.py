import unittest

from simulation.neuroevolution import NeuralPolicyEvolution


class EvolutionProgressTests(unittest.TestCase):
    def test_callback_and_history_include_generation_timing(self):
        progress_updates = []
        trainer = NeuralPolicyEvolution(
            {},
            population_size=2,
            generations=2,
            seeds=(1,),
            progress_callback=progress_updates.append,
            random_seed=1,
        )
        trainer._score = lambda policy: {
            "policy": policy,
            "fitness": 12.5,
            "mean_metrics": {},
        }

        result = trainer.run()

        self.assertEqual(len(progress_updates), 2)
        self.assertEqual(progress_updates, result["history"])
        self.assertEqual(progress_updates[0]["generation_number"], 1)
        self.assertEqual(progress_updates[1]["generation_number"], 2)
        self.assertAlmostEqual(progress_updates[0]["best_fitness"], 12.5)
        self.assertAlmostEqual(progress_updates[0]["mean_fitness"], 12.5)
        self.assertGreaterEqual(progress_updates[0]["generation_time_s"], 0.0)
        self.assertGreaterEqual(progress_updates[1]["elapsed_time_s"], 0.0)
        self.assertGreaterEqual(result["training_time_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
