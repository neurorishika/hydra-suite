def test_fragment_solver_imports():
    from hydra_suite.core.identity.fragment_solver import run_fragment_solver
    assert callable(run_fragment_solver)
