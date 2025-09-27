# utils/formatters.py
"""
Formatting utilities for display
"""

def format_bytes(bytes_value: float) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format duration in seconds to readable string"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m"
    elif minutes > 0:
        return f"{int(minutes)}m {int(secs)}s"
    else:
        return f"{secs:.1f}s"

def format_frequency(freq_hz: float) -> str:
    """Format frequency to appropriate unit"""
    if freq_hz >= 1e12:
        return f"{freq_hz/1e12:.2f} THz"
    elif freq_hz >= 1e9:
        return f"{freq_hz/1e9:.2f} GHz"
    elif freq_hz >= 1e6:
        return f"{freq_hz/1e6:.2f} MHz"
    elif freq_hz >= 1e3:
        return f"{freq_hz/1e3:.2f} kHz"
    else:
        return f"{freq_hz:.2f} Hz"

