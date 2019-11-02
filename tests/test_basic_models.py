from toy import Model


class TestBasicParticleModel:
    time = 1
    steps = 10
    acceleration = 2

    def get_class(self):
        class Particle(Model):
            # Variables
            x = 0, '[m] position'
            v = 0, '[m/s] speed'

            # Equations of motion
            D_x = v
            D_v = self.acceleration

        return Particle

    def get_model(self):
        return self.get_class()()

    def get_run(self):
        return self.get_model().run(0, self.time, self.steps)

    def test_create_model_class(self):
        cls = self.get_class()

    def test_create_model(self):
        model = self.get_model()

        assert set(model.vars) == {'x', 'v'}
        assert model.vars == {'x': model.x, 'v': model.v}
        assert model.params == {}
        assert model.computed_terms == {}

    def _test_model_meta(self):
        meta = self.get_model()._meta
        assert meta.state_size == 2
        assert meta.computed_size == 0
        assert meta.params_size == 0
        assert meta.storage_size == 2

    def test_model_run(self):
        run = self.get_run()
        assert len(run[0]) == self.steps
        assert len(run[1]) == self.steps
        # assert len(run.x_ts) == self.steps
        # assert len(run.v_ts) == self.steps

    def _test_energy_conservation(self):
        model = self.get_run()
        X, V = model.timeseries('x v')
        K = 0.5 * V ** 2
        U = self.acceleration * X
        E = K - U
        assert E.std() <= (K.mean() + abs(U).mean()) * 0.1
        assert X.std() != 0
