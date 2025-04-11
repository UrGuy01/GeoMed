-- Create a simplified symptoms_data table without spatial index
CREATE TABLE IF NOT EXISTS public.symptoms_data (
    id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
    symptoms text[] NOT NULL,
    diagnosis text NOT NULL,
    confidence float NOT NULL,
    timestamp timestamptz NOT NULL DEFAULT now(),
    source text,
    
    -- Location data
    latitude float,
    longitude float,
    accuracy float,
    location_source text,
    
    -- IP-based location data (optional)
    ip_address text,
    city text,
    region text,
    country text,
    
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.symptoms_data ENABLE ROW LEVEL SECURITY;

-- Create policies
-- Allow anonymous insert (for public diagnoses)
CREATE POLICY "Allow anonymous inserts" 
ON public.symptoms_data 
FOR INSERT
TO anon
WITH CHECK (true);

-- Allow anonymous select (for public diagnoses)
CREATE POLICY "Allow anonymous selects" 
ON public.symptoms_data 
FOR SELECT
TO anon
USING (true);

-- Grant necessary permissions
GRANT SELECT, INSERT ON public.symptoms_data TO anon;
GRANT ALL ON public.symptoms_data TO authenticated;

-- Make sure the id sequence is granted
GRANT USAGE, SELECT ON SEQUENCE symptoms_data_id_seq TO anon;
GRANT USAGE, SELECT ON SEQUENCE symptoms_data_id_seq TO authenticated; 