USE zillow;

SELECT properties_2017.*
	FROM properties_2017
		JOIN propertylandusetype USING (propertylandusetypeid)
        JOIN predictions_2017 USING (parcelid)
    WHERE propertylandusedesc = 'Single Family Residential'
		AND predictions_2017.transactiondate LIKE '2017%';