import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface DistributionChartProps {
  distribution: [number, number, number, number, number];
  meanRating?: number;
}

export function DistributionChart({
  distribution,
  meanRating,
}: DistributionChartProps) {
  // Transform data for Recharts
  const data = distribution.map((probability, index) => ({
    rating: index + 1,
    probability: probability * 100, // Convert to percentage
  }));

  // Color gradient from red to green
  const colors = ["#ef4444", "#f59e0b", "#eab308", "#84cc16", "#22c55e"];

  return (
    <>
      {meanRating && (
        <div className="mb-4">
          <p className="text-sm text-slate-600">
            Mean Rating:{" "}
            <span className="font-semibold">{meanRating.toFixed(2)}</span> / 5.0
          </p>
        </div>
      )}
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="rating"
            label={{
              value: "Rating (1-5)",
              position: "insideBottom",
              offset: -5,
            }}
          />
          <YAxis
            label={{
              value: "Probability (%)",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip
            formatter={(value: number) => `${value.toFixed(1)}%`}
            labelFormatter={(label) => `Rating ${label}`}
          />
          <Bar dataKey="probability" radius={[8, 8, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </>
  );
}
