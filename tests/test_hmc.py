import unittest
import math
import time
import hamiltonian_sampler_rs as hmc


class TestHamiltonianSampler(unittest.TestCase):
    """
    Rustで実装されたHMCサンプラーの包括的なテストスイート。
    インターフェースの整合性、アルゴリズムの挙動、エッジケースを検証する。
    """

    def test_01_signature_and_return_types(self):
        """基本機能テスト: 引数を受け取り、正しい型とサイズの戻り値を返すか検証"""
        n_samples = 100
        step_size = 0.1
        num_steps = 10
        start_x = 0.0
        start_y = 0.0
        dist_type = "bimodal"

        # 実行
        samples, acceptance_rate = hmc.sample(
            n_samples, step_size, num_steps, start_x, start_y, dist_type
        )

        # 検証
        self.assertIsInstance(samples, list, "サンプルはリストであるべき")
        self.assertEqual(len(samples), n_samples, "指定したサンプル数が返されるべき")
        self.assertIsInstance(acceptance_rate, float, "採択率はfloatであるべき")

        # サンプルの中身の型検証
        first_sample = samples[0]
        self.assertIsInstance(first_sample, tuple, "各サンプルはタプルであるべき")
        self.assertEqual(len(first_sample), 2, "2次元データ (x, y) であるべき")
        self.assertIsInstance(first_sample[0], float)
        self.assertIsInstance(first_sample[1], float)

        # 採択率の範囲検証 (0.0 <= rate <= 1.0)
        self.assertGreaterEqual(acceptance_rate, 0.0)
        self.assertLessEqual(acceptance_rate, 1.0)

    def test_02_distribution_switching(self):
        """分布切り替えテスト: 'banana' と 'bimodal' が正しく認識されるか"""
        # Banana分布
        _, acc_banana = hmc.sample(50, 0.1, 5, 0.0, 0.0, "banana")
        # 実行できればOK（内部ロジックの違いは統計テスト以外では判別困難だが、クラッシュしないことを確認）
        self.assertIsInstance(acc_banana, float)

        # Bimodal分布
        _, acc_bimodal = hmc.sample(50, 0.1, 5, 0.0, 0.0, "bimodal")
        self.assertIsInstance(acc_bimodal, float)

    def test_03_unknown_distribution_fallback(self):
        """堅牢性テスト: 未知の分布名でもクラッシュせずデフォルト(Bimodal)で動作するか"""
        # Rust実装では `_ => DistType::Bimodal` となっている仕様を確認
        try:
            samples, _ = hmc.sample(10, 0.1, 5, 0.0, 0.0, "unknown_dist_name")
            self.assertEqual(len(samples), 10)
        except Exception as e:
            self.fail(f"未知の分布名を渡してクラッシュしました: {e}")

    def test_04_step_size_sensitivity(self):
        """
        アルゴリズム特性テスト: Step Sizeと採択率の物理的な関係を確認

        仮説:
        - 小さい Step Size (0.01) -> 採択率は高い (ほぼ1.0)
        - 大きい Step Size (0.5)  -> 採択率は低い (誤差が大きいため棄却される)
        """
        n = 200
        steps = 10

        # Case A: Small Step Size (High Acceptance)
        _, rate_high = hmc.sample(n, 0.01, steps, 0.0, 0.0, "bimodal")

        # Case B: Large Step Size (Low Acceptance)
        _, rate_low = hmc.sample(n, 2.2, steps, 0.0, 0.0, "bimodal")

        print(
            f"\n[Sensitivity Test] Small Step(0.01): {rate_high:.2%}, Large Step(2.2): {rate_low:.2%}"
        )

        # 検証: 小さいステップの採択率 > 大きいステップの採択率
        self.assertGreater(
            rate_high,
            rate_low,
            "ステップサイズが小さい方が採択率が高くなるべき（エネルギー誤差が小さいため）",
        )

        # 具体的な閾値チェック
        self.assertGreater(
            rate_high, 0.8, "ステップサイズ0.01なら採択率は非常に高いはず"
        )
        self.assertLess(rate_low, 0.5, "ステップサイズ0.8なら採択率は低いはず")

    def test_05_sampling_movement(self):
        """動作テスト: サンプルが初期位置から移動しているか（分散があるか）"""
        n = 100
        # 初期位置 (0,0)
        samples, _ = hmc.sample(n, 0.15, 20, 0.0, 0.0, "bimodal")

        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]

        # 全ての点が (0,0) のままでないことを確認
        var_x = sum((x - 0) ** 2 for x in xs) / n
        var_y = sum((y - 0) ** 2 for y in ys) / n

        self.assertGreater(var_x, 0.001, "X座標が移動していません")
        self.assertGreater(var_y, 0.001, "Y座標が移動していません")

    def test_06_performance_stress(self):
        """パフォーマンステスト: 比較的大量のサンプル生成"""
        n_stress = 50_000
        start_time = time.time()

        # 5万サンプル生成
        samples, _ = hmc.sample(n_stress, 0.1, 5, 0.0, 0.0, "bimodal")

        duration = time.time() - start_time
        print(f"\n[Stress Test] Generated {n_stress} samples in {duration:.4f} sec")

        self.assertEqual(len(samples), n_stress)
        # Rustならこれくらいは数秒以内で終わるはず（目安: 1秒以内）
        self.assertLess(
            duration, 2.0, "処理時間が遅すぎます（Rustの利点が出ていません）"
        )


if __name__ == "__main__":
    unittest.main()
