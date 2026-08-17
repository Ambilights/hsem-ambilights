# ENTSO-E Price Backup

HSEM can use a pair of ENTSO-E day-ahead price sensors when the primary price
integration has not published a complete delivery day. The primary source
retains priority. ENTSO-E is a redundant publication source, not a price
forecast, and HSEM never adds currency conversion, VAT, tariffs, or fees.

This guide uses the HACS
[`JaccoR/hass-entso-e`](https://github.com/JaccoR/hass-entso-e) integration.
The examples describe version 0.7.5; check the upstream release notes if a
later version changes its price calculation or entity attributes.

## Source contract

Configure two ENTSO-E entries: one for the final import price and one for the
final export price. In HSEM, select each entry's **Average electricity price**
sensor. That sensor exposes the `prices`, `prices_today`, and
`prices_tomorrow` arrays HSEM needs; the scalar current-price sensor does not.

Depending on the integration and Home Assistant version, its entity ID may use
either the `sensor.*` or `entsoe.*` domain; HSEM accepts both.

Both sensors must:

- use the same timezone-aware timestamps and configured cadence;
- cover a complete local delivery day before HSEM can use that day;
- report finite prices in the same currency/kWh unit as the primary pair; and
- already include every source-specific conversion and price adjustment.

HSEM selects sources atomically per slot. It never combines primary import
with ENTSO-E export, or the reverse. A valid primary pair, including genuine
zero or negative prices, always wins. A backup day must contain 92, 96, or 100
quarter-hour points on a Stockholm spring-forward, normal, or fall-back day,
respectively.

The backup covers a publication gap while the primary live entities remain
healthy. It does not bypass HSEM's current-slot live-price outage hold.

## Match the primary price basis

ENTSO-E publishes EUR/MWh. With `energy_scale` set to `kWh`, version 0.7.5
first divides by 1000, evaluates the price modifier, and then multiplies the
result by `1 + VAT`.

Its currency field changes the unit label only; it does not perform foreign
exchange conversion. Its VAT field accepts a fraction: enter `0.25` for 25%,
not `25`.

For a primary Nord Pool setup where:

- export is raw spot converted to SEK; and
- import is that value plus 25% VAT and a `0.824 SEK/kWh` tariff,

the matching formulas are:

```text
export = raw × FX
import = (raw × FX + 0.6592) × 1.25
       = raw × FX × 1.25 + 0.824
```

`0.6592` is the tariff before ENTSO-E applies VAT:
`0.6592 × 1.25 = 0.824`. Adding `0.824` inside the import modifier would
produce the wrong result.

Use these settings for the two entries:

| Setting | Import entry | Export entry |
|---|---:|---:|
| Delivery area | Same as primary | Same as primary |
| Period | `PT15M` | `PT15M` |
| Energy scale | `kWh` | `kWh` |
| Calculation mode | `publish` | `publish` |
| Currency label | Primary currency, for example `SEK` | Primary currency, for example `SEK` |
| VAT | `0.25` | `0` |

Import modifier:

```jinja
{% set fx = states('input_number.ecb_eur_sek_rate') | float(8) %}{{ ((current_price * fx) + 0.6592) if 8 < fx < 20 else 'invalid' }}
```

Export modifier:

```jinja
{% set fx = states('input_number.ecb_eur_sek_rate') | float(8) %}{{ (current_price * fx) if 8 < fx < 20 else 'invalid' }}
```

The bounds deliberately make the modifiers fail closed before the exchange
rate helper has received its first valid value. Adapt the conversion, VAT, and
tariff to the primary sensors when their price basis differs from this example.

## Durable ECB exchange-rate helper

The official ECB series `EXR.D.SEK.EUR.SP00.A` reports SEK per EUR. A plain
Home Assistant REST sensor becomes unavailable when its request fails and does
not restore its payload after a failed startup request. Store the last valid
observation in a restoring `input_number` so weekends and temporary ECB
outages do not turn valid ENTSO-E prices into zero or an invalid conversion.

The following can live in a Home Assistant package. Do not add an `initial`
value: Home Assistant then restores the helper's previous state. On its first
start the helper uses its minimum, so wait until it contains a real value above
8 before creating or reloading the ENTSO-E entries.

```yaml
input_number:
  ecb_eur_sek_rate:
    name: ECB EUR/SEK rate
    min: 8
    max: 20
    step: 0.0001
    mode: box
    unit_of_measurement: "SEK/EUR"

rest:
  - resource: "https://data-api.ecb.europa.eu/service/data/EXR/D.SEK.EUR.SP00.A"
    params:
      format: jsondata
      lastNObservations: 1
    headers:
      Accept: application/json
    timeout: 30
    scan_interval: 21600
    sensor:
      - name: ECB EUR SEK Latest
        unique_id: ecb_eur_sek_latest
        unit_of_measurement: "SEK/EUR"
        state_class: measurement
        availability: >
          {{ value_json.dataSets is defined
             and value_json.dataSets
             and value_json.dataSets[0].series is defined
             and value_json.dataSets[0].series | count > 0 }}
        value_template: >
          {% set series =
             value_json.dataSets[0].series.values() | list | first %}
          {% set observation =
             series.observations.values() | list | first %}
          {{ observation[0] | float }}

automation:
  - id: store_latest_valid_ecb_eur_sek_rate
    alias: Store latest valid ECB EUR/SEK rate
    mode: restart
    triggers:
      - trigger: state
        entity_id: sensor.ecb_eur_sek_latest
      - trigger: homeassistant
        event: start
      - trigger: time_pattern
        hours: "/6"
    conditions:
      - condition: template
        value_template: >
          {% set fx = states('sensor.ecb_eur_sek_latest') | float(0) %}
          {{ 8 < fx < 20 }}
    actions:
      - action: input_number.set_value
        target:
          entity_id: input_number.ecb_eur_sek_rate
        data:
          value: "{{ states('sensor.ecb_eur_sek_latest') | float }}"
```

The retained value can become stale during a prolonged ECB outage. Monitor the
raw REST sensor and alert if no fresh observation arrives for several business
days. Do not substitute a fixed exchange rate silently.

## Validation before enabling the backup

Compare overlapping timestamps in Home Assistant Developer Tools before
selecting the sensors in HSEM:

1. Confirm both average-price sensors expose `prices_today` and, after market
   publication, a complete `prices_tomorrow` at the configured cadence.
2. Confirm their declared unit exactly matches the corresponding primary unit.
3. For every overlapping slot, verify approximately:
   `import = export × 1.25 + 0.824` for the example above.
4. Compare ENTSO-E export against primary export. Small differences can occur
   when their currency conversions use different fixing times; large or
   systematic differences indicate a wrong modifier.
5. Confirm negative prices survive the conversion and that the import tariff
   is applied exactly once.
6. Reopen HSEM's **Electricity Prices** step and select both average-price
   sensors together. The config flow rejects a partial pair, wrong units,
   misaligned timestamps, or a cadence mismatch.
7. Inspect `entsoe_price_backup_status` after the next update. An incomplete
   delivery day remains unused and reports a `rejection_reason`; an accepted
   day increases `matched_slots`.
8. Inspect slot provenance. Primary slots should remain labelled `primary`;
   only eligible publication gaps should use `entsoe`.

## Version 0.7.5 caveats

- The currency option is a label, not a conversion.
- VAT is a fraction and is applied after the modifier.
- Only the average-price sensor carries the timestamped price arrays.
- The integration's own availability check uses a fixed threshold of more than
  20 points. At `PT15M`, that is not proof of a complete delivery day; HSEM
  independently requires complete aligned backup days before using them.
- The day-ahead request uses document type `A44` and matching in/out domains,
  but does not send `contract_MarketAgreement.type=A01`. Treat unexpected or
  mixed market documents as a source problem and inspect the arrays carefully.
- Its piecewise-constant (`A03`) parser extends each point only until the last
  point present in a period. A malformed or truncated tail can therefore look
  plausible but remain incomplete; HSEM rejects incomplete days.
- Debug logging includes the request parameter mapping, which contains the
  ENTSO-E security token. Do not enable or share upstream debug logs without
  redacting credentials.

Keep the ENTSO-E integration current and revalidate these assumptions after an
upgrade.
