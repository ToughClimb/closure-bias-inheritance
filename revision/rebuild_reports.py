"""Rebuild numerical summaries and figures from explicit, read-only inputs."""
from project_paths import ROOT, INPUT_ROOT


def main():
    import build_report
    import build_extended_report
    import analyze_error_geometry
    import audit_initial_conditions
    build_report.main()
    build_extended_report.extended()
    analyze_error_geometry.main()
    audit_initial_conditions.main()
    print('rebuilt reports in',ROOT,'from',INPUT_ROOT)


if __name__=='__main__':main()
