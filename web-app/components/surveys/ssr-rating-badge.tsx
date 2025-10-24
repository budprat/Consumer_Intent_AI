import { Badge } from "@/components/ui/badge";

interface SSRRatingBadgeProps {
  rating: number;
  showLabel?: boolean;
  size?: "default" | "large";
}

export function SSRRatingBadge({
  rating,
  showLabel = true,
  size = "default",
}: SSRRatingBadgeProps) {
  // Color based on rating (1-5 scale)
  const getColor = (rating: number) => {
    if (rating >= 4) return "bg-green-100 text-green-800 border-green-300";
    if (rating >= 3) return "bg-blue-100 text-blue-800 border-blue-300";
    if (rating >= 2) return "bg-yellow-100 text-yellow-800 border-yellow-300";
    return "bg-red-100 text-red-800 border-red-300";
  };

  const sizeClasses = size === "large" ? "text-2xl px-4 py-2" : "";

  return (
    <Badge variant="outline" className={`${getColor(rating)} ${sizeClasses}`}>
      {showLabel && "SSR: "}
      {rating.toFixed(1)} / 5.0
    </Badge>
  );
}
