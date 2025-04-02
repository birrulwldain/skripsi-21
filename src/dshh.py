from manim import *
import numpy as np
from dsh import DataFetcher, SpectrumSimulator  # Pastikan Anda telah mengimport modul yang dibutuhkan

class SpectrumAnimation(Scene):
    def construct(self):
        # Ambil data dari database
        data_fetcher = DataFetcher(db_nist="data1.db", db_spectrum="tanah_vulkanik.db")
        nist_data = data_fetcher.get_nist_data("Ca", 2)  # Fe ion stage 1

        # Buat ValueTracker untuk suhu
        temperature = ValueTracker(100)

        # Buat axes
        axes = Axes(
            x_range=[200, 900, 50],
            y_range=[0, 1.1, 0.1],
            axis_config={"include_numbers": True},
        ).add_coordinates()
        x_label = axes.get_x_axis_label(r"Panjang Gelombang (nm)", direction=RIGHT)
        y_label = axes.get_y_axis_label(r"Intensitas (a.u.)", direction=UP)

        # Fungsi untuk mengupdate grafik spektrum
        def update_spectrum(mob):
            simulator = SpectrumSimulator(nist_data, temperature.get_value())
            wavelengths, intensities = simulator.simulate()
            new_graph = axes.plot(lambda x: np.interp(x, wavelengths, intensities), color=BLUE)
            mob.become(new_graph)

        # Grafik spektrum awal
        simulator = SpectrumSimulator(nist_data, temperature.get_value())
        wavelengths, intensities = simulator.simulate()
        graph = axes.plot(lambda x: np.interp(x, wavelengths, intensities), color=BLUE)

        # Label suhu
        temp_label = always_redraw(
            lambda: MathTex(f"T = {temperature.get_value():.0f} K").move_to(UP * 3)
        )

        # Tambahkan elemen ke scene
        self.add(axes, x_label, y_label, graph, temp_label)

        # Animasikan perubahan suhu dan spektrum
        self.play(
            temperature.animate.set_value(10000),
            UpdateFromFunc(graph, update_spectrum),
            run_time=10,
            rate_func=linear,
        )
        self.wait()