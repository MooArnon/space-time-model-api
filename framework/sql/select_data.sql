with scope as (
    select
        scraped_timestamp
        ,price
    from warehouse.fact_raw_data
    where asset = '<ASSET>'
    order by scraped_timestamp desc
    limit 110
)
select *
from scope
order by scraped_timestamp asc
;