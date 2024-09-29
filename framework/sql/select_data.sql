with scope as (
    select
        CAST(scraped_timestamp AS timestamp(3)) as scraped_timestamp
        ,price
        ,asset
    from fact_raw_data
    where asset = '<ASSET>'
    order by scraped_timestamp desc
    limit 50000
)
select *
from scope
order by scraped_timestamp asc
limit 50000
;