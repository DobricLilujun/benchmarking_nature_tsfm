"""Compatibility wrapper.

This script keeps the old entry point used by run_all_*.sh, but delegates to the
new reusable CLI in real_v_tsfm.
"""

from real_v_tsfm.cli.extract import main


if __name__ == "__main__":
    main()