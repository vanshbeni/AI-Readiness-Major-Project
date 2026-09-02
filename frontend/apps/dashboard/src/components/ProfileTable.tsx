import React from 'react';
import { ColumnProfile } from '../services/api';
import { Columns, Hash, AlignLeft, Calendar, ToggleLeft, Key } from 'lucide-react';

interface ProfileTableProps {
  columns: ColumnProfile[];
}

export const ProfileTable: React.FC<ProfileTableProps> = ({ columns }) => {
  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'numeric': return <Hash size={13} color="#38bdf8" />;
      case 'categorical': return <AlignLeft size={13} color="#a855f7" />;
      case 'boolean': return <ToggleLeft size={13} color="#10b981" />;
      case 'datetime': return <Calendar size={13} color="#f59e0b" />;
      case 'id': return <Key size={13} color="#94a3b8" />;
      default: return <AlignLeft size={13} color="#94a3b8" />;
    }
  };

  return (
    <div className="glass-panel" style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Columns size={18} color="#38bdf8" />
          <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Feature Profiling Matrix</h3>
        </div>
        <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)' }}>
          {columns.length} Features Ingested
        </span>
      </div>

      <div style={{ overflowX: 'auto', borderRadius: 8, border: '1px solid var(--border)' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.825rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255, 255, 255, 0.04)', borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Column Name</th>
              <th style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Inferred Type</th>
              <th style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Missingness</th>
              <th style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Uniqueness</th>
              <th style={{ padding: '0.65rem 0.85rem', fontWeight: 600, color: 'var(--text-muted)' }}>Distribution Summary</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col) => (
              <tr
                key={col.name}
                style={{
                  borderBottom: '1px solid rgba(255, 255, 255, 0.04)',
                  background: col.is_target ? 'rgba(56, 189, 248, 0.04)' : 'transparent',
                }}
              >
                {/* Column Name */}
                <td style={{ padding: '0.65rem 0.85rem', fontWeight: 600 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ color: col.is_target ? '#38bdf8' : 'var(--text-main)' }}>{col.name}</span>
                    {col.is_target && (
                      <span className="badge badge-medium" style={{ fontSize: '0.65rem', padding: '0.1rem 0.35rem' }}>
                        Target
                      </span>
                    )}
                  </div>
                </td>

                {/* Inferred Type */}
                <td style={{ padding: '0.65rem 0.85rem' }}>
                  <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', padding: '0.2rem 0.5rem', borderRadius: 6, background: 'rgba(255, 255, 255, 0.04)', border: '1px solid var(--border)', fontSize: '0.75rem', textTransform: 'capitalize' }}>
                    {getTypeIcon(col.inferred_type)}
                    <span>{col.inferred_type}</span>
                  </div>
                </td>

                {/* Missing % */}
                <td style={{ padding: '0.65rem 0.85rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <div style={{ width: 45, height: 5, background: 'rgba(255, 255, 255, 0.1)', borderRadius: 3, overflow: 'hidden' }}>
                      <div
                        style={{
                          width: `${col.missing_pct}%`,
                          height: '100%',
                          background: col.missing_pct > 20 ? '#f43f5e' : col.missing_pct > 0 ? '#f59e0b' : '#10b981',
                        }}
                      />
                    </div>
                    <span style={{ color: col.missing_pct > 0 ? '#fbbf24' : 'var(--text-dim)', fontWeight: col.missing_pct > 0 ? 600 : 400 }}>
                      {col.missing_count} ({col.missing_pct}%)
                    </span>
                  </div>
                </td>

                {/* Unique */}
                <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-muted)' }}>
                  {col.unique_count} ({col.unique_pct}%)
                </td>

                {/* Distribution Summary */}
                <td style={{ padding: '0.65rem 0.85rem', color: 'var(--text-dim)', fontSize: '0.75rem', fontFamily: 'monospace' }}>
                  {col.numeric_stats ? (
                    <span>
                      μ={col.numeric_stats.mean} | med={col.numeric_stats.median} | min={col.numeric_stats.min} | max={col.numeric_stats.max} | skew={col.numeric_stats.skewness}
                    </span>
                  ) : col.categorical_stats && col.categorical_stats.top_categories.length > 0 ? (
                    <span>
                      Top: {col.categorical_stats.top_categories.slice(0, 3).map(c => `${c.value} (${c.percentage}%)`).join(', ')}
                    </span>
                  ) : (
                    <span>—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
