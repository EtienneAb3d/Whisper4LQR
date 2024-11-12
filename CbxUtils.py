def format_time_ms(seconds,trunk:bool = False):
    # Separate whole seconds and milliseconds
    whole_seconds, milliseconds = divmod(seconds, 1)
    
    # Convert to integers
    whole_seconds = int(whole_seconds)
    milliseconds = int(milliseconds * 1000)
    
    # Calculate minutes and remaining seconds
    minutes, seconds = divmod(whole_seconds, 60)
    
    if trunk and minutes <= 0:
        return f"{seconds:02d}:{milliseconds:03d}"

    # Format the string
    return f"{minutes:02d}:{seconds:02d}:{milliseconds:03d}"

