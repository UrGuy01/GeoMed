-- Create symptoms_data table for storing location-tagged symptom data
CREATE TABLE IF NOT EXISTS public.symptoms_data (
    id uuid DEFAULT uuid_generate_v4() PRIMARY KEY,
    symptoms text[] NOT NULL,  -- Array of symptom strings
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

-- Allow authenticated users to select all data
CREATE POLICY "Allow authenticated read access" 
ON public.symptoms_data 
FOR SELECT
TO authenticated
USING (true);

-- Create index on timestamp for efficient time-range queries
CREATE INDEX IF NOT EXISTS symptoms_data_timestamp_idx ON public.symptoms_data (timestamp);

-- Create index on diagnosis for filtering by disease
CREATE INDEX IF NOT EXISTS symptoms_data_diagnosis_idx ON public.symptoms_data (diagnosis);

-- Create GIN index on symptoms array for efficient symptom filtering
CREATE INDEX IF NOT EXISTS symptoms_data_symptoms_idx ON public.symptoms_data USING GIN (symptoms);

-- Create spatial index for location queries
CREATE INDEX IF NOT EXISTS symptoms_data_location_idx ON public.symptoms_data USING gist (
    ST_SetSRID(ST_Point(longitude, latitude), 4326)
) WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Grant necessary permissions
GRANT SELECT ON public.symptoms_data TO anon;
GRANT INSERT ON public.symptoms_data TO anon;
GRANT ALL ON public.symptoms_data TO authenticated; 