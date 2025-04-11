import React from 'react';
import { Box, TextField } from '@mui/material';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';

const DateRangePicker = ({ dateRange, setDateRange }) => {
  const handleStartDateChange = (date) => {
    setDateRange(prev => ({
      ...prev,
      startDate: date ? date.toISOString().split('T')[0] : null
    }));
  };

  const handleEndDateChange = (date) => {
    setDateRange(prev => ({
      ...prev,
      endDate: date ? date.toISOString().split('T')[0] : null
    }));
  };

  return (
    <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <DatePicker
          label="Start Date"
          value={dateRange.startDate}
          onChange={handleStartDateChange}
          renderInput={(params) => <TextField {...params} />}
        />
        <DatePicker
          label="End Date"
          value={dateRange.endDate}
          onChange={handleEndDateChange}
          renderInput={(params) => <TextField {...params} />}
        />
      </LocalizationProvider>
    </Box>
  );
};

export default DateRangePicker; 